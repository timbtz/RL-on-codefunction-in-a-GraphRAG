"""IngestSearchRewardAdapter -- score ONE candidate that spans BOTH the TS
graph-ingestion code AND the Python search code (Phase-2 co-optimize).

The seam is identical to ``McqRewardAdapter.score`` (the loop calls
``self._reward.score(fn, idxs, src=..., return_rows=..., per_query_timeout_s=...)``
and nothing else), so swapping it in is a one-line wiring change in
``campaign.boot_search``. What differs is a TWO-PHASE rollout hidden behind that
seam (see ``Orchestration/Plans/cooptimize-CONTRACT.md``):

  (A) HASH + CACHE -- hash ONLY the ingestion ``.ts`` overlay file(s); the graph
      database name is ``f"{cfg.ingest_db_prefix}{ingest_hash}"``. Neo4j Community
      is SINGLE-DB, so caching is wipe-and-rebuild of the default database keyed by
      a "currently-loaded hash" marker; a search-only edit (same .ts) reuses the
      built graph. Ingestion is SERIALIZED (an in-process lock + a cross-process
      file lock) so concurrent candidates never clobber the shared DB.
  (B) INGEST -- materialize the ``extract.ts`` overlay into the graphmod tree and
      subprocess ``npx tsx ... ingest.ts``. A nonzero exit / timeout / tsx build
      error -> a CRASHED MetricVector and (with return_rows) rows whose ``error``
      carries the tsx stderr, so the bad ingestion edit is REJECTED + self-repaired.
  (C) SEARCH + SCORE -- a thin Target whose ``.run(file_set, queries, timeout_s)``
      (the SAME contract ``McqReward`` consumes) imports the candidate ``search.py``
      from the overlay, runs ``SearchService`` over each query against the built
      graph, and returns per-query context + cost. We DELEGATE to
      ``McqReward.score(...)`` so the judge-free answerer + retrieval_hit + usd are
      byte-identical to the graph_search path.
  (D) COST AXES -- ``mv.ingest_usd`` / ``mv.ingest_tokens`` from ``ingest_cost.json``,
      ``mv.n_queries = len(idxs)``; ``mv.sanitize()`` (``total_usd_per_query`` is then
      automatic).
  (E) ATTRIBUTION -- for each MISSED row, a read-only graph probe
      (``optimizer.graph_inspect.GraphInspector``) classifies the miss as
      NOT_INGESTED / ORPHANED / UNREACHABLE / RANKING and attaches it as
      ``row["graph_attribution"]`` (``fast_loop._failure_record`` prefers that key).
      The gold node id is the ONLY gold signal the probe sees -- never the options.

``fn`` is a ``FileSet`` whose ``base_dir`` is the monorepo root and whose overlay
holds the two editable files at ``cfg.ingest_editable_files`` relpaths
(``graphmod/src/ingestion/extract.ts`` + ``graphsearch/src/search/search.py``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Optional

from graphretr_opt.reward.objectives import MetricVector

# The search-target contract types (context + CostMeter + error) that McqReward
# consumes verbatim. Reused so the seam matches SubprocessSearchTarget exactly.
from graphretr_opt.env.search_target import CostMeter, SearchResult
from graphretr_opt.optimizer.graph_inspect import GraphInspector

# MCQ quality keys live in the sibling graphsearch package (put on sys.path by
# campaign.boot_search, like McqReward itself). Import lazily-tolerantly so this
# module is importable even before that path is wired (e.g. py_compile / probes).
try:  # pragma: no cover - import-path resilience
    from graphsearch.reward.qa_objectives import MCQ_PRIMARY, MCQ_QUALITY_KEYS
    from graphsearch.reward import McqRewardAdapter as _McqRewardAdapter
except Exception:  # noqa: BLE001 - fall back to the contract constants
    MCQ_PRIMARY = "mcq_accuracy"
    MCQ_QUALITY_KEYS = ("mcq_accuracy", "retrieval_hit")
    _McqRewardAdapter = None


# A process-wide ingestion lock: ALL IngestSearchRewardAdapter instances in this
# process serialize their ingest+swap against the shared single Neo4j DB.
_INGEST_LOCK = threading.Lock()


def _reflect_row(mcq_row: dict) -> dict:
    """McqReward row -> FastLoop._reflect/_failure_record shape. Reuses
    ``McqRewardAdapter._to_reflect_row`` (the canonical translation) when the
    graphsearch package is importable; otherwise a faithful local copy so the
    adapter never silently drifts from the graph_search path."""
    if _McqRewardAdapter is not None:
        return _McqRewardAdapter._to_reflect_row(mcq_row)
    rh = 1.0 if mcq_row.get("retrieval_hit") else 0.0
    ob = 1.0 if mcq_row.get("openbook_correct") else 0.0
    has_ctx = bool(mcq_row.get("context_preview")) and not mcq_row.get("error")
    return {
        "idx": mcq_row["idx"], "q_id": mcq_row.get("q_id"),
        "query": mcq_row.get("question"), "answer_ids": [],
        "metrics": {"recall@20": rh, "hit@1": ob, "hit@5": ob, "mrr": ob},
        "retrieved": [] if not has_ctx else [(0, 1.0)],
        "gold_ranks": {}, "recall@100": rh, "error": mcq_row.get("error"),
        "chosen_idx": mcq_row.get("chosen_idx"), "gold_idx": mcq_row.get("gold_idx"),
        "openbook_correct": bool(mcq_row.get("openbook_correct")),
        "closedbook_correct": bool(mcq_row.get("closedbook_correct")),
        "retrieval_hit": bool(mcq_row.get("retrieval_hit")),
        "context_preview": mcq_row.get("context_preview"),
    }


# --------------------------------------------------------------------------- #
# Counting graph proxy (per-query db_load) + citation parsing                  #
# --------------------------------------------------------------------------- #
class _CountingGraph:
    """Transparent proxy over the injected Neo4jGraph that counts ``.query()``
    calls, so the search phase can report ``db_queries`` per query without the
    candidate ``search.py`` needing to meter anything (mirrors qa_runner's
    ``_MeteredGraph``). Read-only pass-through -- it never alters the query."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.db_queries = 0

    def query(self, cypher: str, params: Optional[dict] = None):
        self.db_queries += 1
        return self._inner.query(cypher, params or {})

    def __getattr__(self, name):  # forward .schema, .refresh_schema, etc.
        return getattr(self._inner, name)


def _ids_from_citations(citations: list) -> list:
    """``['[doc:auth]', '[ticket:PROJ-1]']`` -> ``['doc:auth', 'ticket:PROJ-1']``.
    The search returns bracketed citation tokens; the attribution probe wants the
    bare canonical ids."""
    out = []
    for c in citations or []:
        s = str(c).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if s:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# The thin search Target (matches the McqReward .run contract)                 #
# --------------------------------------------------------------------------- #
class _IngestSearchTarget:
    """``run(file_set, queries, timeout_s) -> {query: SearchResult}`` -- the SAME
    contract ``McqReward`` consumes (see ``env/search_target.SearchTarget`` and
    ``FakeSearchTarget``). It imports the candidate ``search.py`` from the overlay
    via importlib, constructs a Neo4j graph scoped to the built ``graph_<hash>``
    database, runs ``SearchService`` over each query, and shapes the result into
    the context/cost/error triple the reward reads.

    Per-query diagnostics (``seed-ish retrieved ids``) are stashed on
    ``self.diagnostics[query]`` so the adapter's attribution probe can tell a
    RANKING miss (gold was retrieved) from an UNREACHABLE one.
    """

    def __init__(self, graphdb: str, cfg: Any,
                 search_relpath: str) -> None:
        self._graphdb = graphdb
        self._cfg = cfg
        self._search_relpath = search_relpath
        self.diagnostics: dict = {}

    # -- candidate import ------------------------------------------------- #
    def _load_search_module(self, file_set):
        """Materialize the overlay ``search.py`` to a temp file and import it via
        importlib (a fresh module object per call -- no cross-candidate caching).
        ``search.py`` is a standalone module (no intra-package imports), so a
        file-location import is sufficient."""
        src = file_set.overlay[FileSet._norm(self._search_relpath)]
        tmpdir = tempfile.mkdtemp(prefix="ingest_search_py_")
        try:
            path = os.path.join(tmpdir, "candidate_search.py")
            with open(path, "w") as f:
                f.write(src)
            modname = f"candidate_search_{abs(hash(src)) & 0xffffffff:x}"
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            # Register BEFORE exec: a module that uses ``@dataclass`` under
            # ``from __future__ import annotations`` (search.py does both) makes
            # dataclasses resolve string annotations via ``sys.modules[__module__]``
            # -- absent that entry it raises AttributeError mid-import.
            sys.modules[modname] = mod
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            finally:
                sys.modules.pop(modname, None)
            return mod
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # -- Neo4j graph (deferred import; no DB at module import time) -------- #
    def _build_graph(self):
        """Construct a ``langchain_neo4j.Neo4jGraph`` scoped to the candidate's
        ``graph_<hash>`` database, wrapped in the counting proxy. Imported lazily
        so this module loads without langchain_neo4j installed."""
        from langchain_neo4j import Neo4jGraph  # deferred: needs the dep + a DB
        cfg = self._cfg
        graph = Neo4jGraph(
            url=cfg.neo4j_url,
            username=cfg.neo4j_user,
            password=cfg.neo4j_password,
            database=self._graphdb,
        )
        return _CountingGraph(graph)

    def run(self, file_set, queries, timeout_s: float) -> dict:
        """Run every query through the candidate ``SearchService`` against the
        built graph. A build/import failure errors ALL queries (scored as misses,
        never fatal); a per-query crash errors just that query."""
        queries = list(queries)
        # A build/import failure dooms every query but must not crash the loop.
        try:
            mod = self._load_search_module(file_set)
            graph = self._build_graph()
            service_kwargs = self._service_cfg(file_set)
            service = mod.SearchService(graph, service_kwargs)
        except Exception as e:  # noqa: BLE001
            msg = f"search build failed: {type(e).__name__}: {e}"
            return {q: SearchResult(context="", cost=CostMeter(), error=msg)
                    for q in queries}

        out: dict = {}
        for q in queries:
            graph.db_queries = 0
            t0 = time.time()
            try:
                res = service.search(q)
                ctx = getattr(res, "context", "") or ""
                citations = getattr(res, "citations", []) or []
                usd = float(getattr(res, "usd", 0.0) or 0.0)
                tokens = int(getattr(res, "tokens", 0) or 0)
                self.diagnostics[q] = {
                    "retrieved_ids": _ids_from_citations(citations),
                    "nodes_seen": int(getattr(res, "nodes_seen", 0) or 0),
                }
                cost = CostMeter(
                    db_queries=graph.db_queries,
                    latency_s=time.time() - t0,
                    usd_cost=usd,
                    # the zero-LLM seed has tokens=0; an enabled LLM hook meters
                    # into search.py's SearchResult.tokens -> charge it as input.
                    tokens_in=tokens,
                    llm_calls=1 if usd > 0.0 else 0,
                )
                out[q] = SearchResult(context=ctx, cost=cost, error=None)
            except Exception as e:  # noqa: BLE001 - per-query crash -> a miss
                out[q] = SearchResult(
                    context="",
                    cost=CostMeter(db_queries=graph.db_queries,
                                   latency_s=time.time() - t0),
                    error=f"{type(e).__name__}: {e}")
        return out

    def _service_cfg(self, file_set) -> dict:
        """Tunable overrides handed to ``SearchService(graph, cfg)``. The seed
        reads everything off its ``DEFAULT_*`` constants, so an empty dict keeps
        the candidate fully in control of its own tunables (the optimizer edits
        them inside ``search.py``). Kept as a hook for future cfg-driven levers."""
        return {}


# Local import so the module loads even if the artifact package moves; FileSet is
# only needed for the ``_norm`` relpath normalization on overlay lookups.
from graphretr_opt.artifact.file_set import FileSet  # noqa: E402


# --------------------------------------------------------------------------- #
# The reward adapter                                                            #
# --------------------------------------------------------------------------- #
class IngestSearchRewardAdapter:
    """Two-phase (ingest -> search -> score) reward with the McqRewardAdapter
    seam. Constructed like ``McqRewardAdapter`` (target unused here -- this
    adapter builds its own per-candidate target), plus the cfg it needs to drive
    ``tsx`` ingestion and the Neo4j graph."""

    def __init__(self, target_unused, mcq_reward, substrate, cfg,
                 default_timeout_s: float = 1800.0,
                 ingest_timeout_s: float = 1800.0) -> None:
        self._target_unused = target_unused      # mirror McqRewardAdapter ctor shape
        self._mcq = mcq_reward
        self._sub = substrate
        self._cfg = cfg
        self._default_timeout_s = float(default_timeout_s)
        self._ingest_timeout_s = float(ingest_timeout_s)
        # In-process "currently-loaded hash" marker (the single shared DB holds at
        # most one graph at a time). The on-disk marker (runs/) survives restarts.
        self._loaded_hash: Optional[str] = None
        self._marker_path = os.path.join(
            getattr(cfg, "runs_dir", os.path.join(cfg.root, "runs")),
            "ingest_loaded.json")
        self._lock_path = os.path.join(
            getattr(cfg, "runs_dir", os.path.join(cfg.root, "runs")),
            "ingest.lock")

    # ================================================================== #
    # The seam                                                            #
    # ================================================================== #
    def score(self, fn, idxs, src=None, return_rows=False,
              per_query_timeout_s=None):
        """fn: the candidate FileSet (overlay = extract.ts + search.py).
        -> MetricVector (or (MetricVector, rows) when return_rows)."""
        file_set = fn
        idxs = list(idxs)
        timeout = (per_query_timeout_s if per_query_timeout_s is not None
                   else self._default_timeout_s)

        # (A) hash ONLY the ingestion .ts overlay -> cache key. The PHYSICAL Neo4j
        # database is always cfg.neo4j_database (Community is single-DB): the hash
        # keys the wipe-and-rebuild cache (loaded-hash marker), it is NOT a real DB
        # name. (Enterprise multi-db isolation -- a distinct DB per hash -- is the
        # documented future hook in _ensure_graph.)
        ingest_hash = self._ingest_hash(file_set)
        graphdb = self._cfg.neo4j_database or "neo4j"

        # (A)+(B) ensure the graph for this hash is built+loaded (serialized,
        # cached). Returns (error_or_None, ingest_cost{tokens, usd}).
        ingest_err, ingest_cost = self._ensure_graph(
            file_set, ingest_hash, graphdb)

        if ingest_err is not None:
            # A bad ingestion edit (tsx build error / nonzero exit / timeout):
            # CRASHED vector + rows carrying the stderr for self-repair.
            mv = self._crashed_vector(len(idxs), ingest_cost)
            if not return_rows:
                return mv
            return mv, self._crashed_rows(idxs, ingest_err)

        # (C) search + score: delegate to McqReward (identical answerer/usd path).
        target = _IngestSearchTarget(
            graphdb, self._cfg, self._search_relpath())
        mv, mcq_rows = self._mcq.score(
            target, file_set, idxs, self._sub, timeout, return_rows=True)

        # (D) cost axes: amortized total_usd_per_query is then automatic.
        mv.ingest_usd = float(ingest_cost.get("usd", 0.0))
        mv.ingest_tokens = int(ingest_cost.get("tokens", 0))
        mv.n_queries = len(idxs)
        mv.sanitize()

        if not return_rows:
            return mv

        rows = [_reflect_row(r) for r in mcq_rows]
        # (E) attribution for missed queries (read-only graph probe).
        self._attribute(rows, idxs, graphdb, target)
        return mv, rows

    # ================================================================== #
    # (A) ingest hash                                                     #
    # ================================================================== #
    def _ingest_files(self) -> tuple:
        """The ingestion-side editable relpaths (the ``.ts`` files) -- the ONLY
        inputs to the ingest hash, so a search-only edit reuses the cached graph."""
        return tuple(FileSet._norm(p) for p in self._cfg.ingest_editable_files
                     if str(p).endswith(".ts"))

    def _search_relpath(self) -> str:
        """The search-side editable relpath (the ``.py`` file the search phase
        imports)."""
        for p in self._cfg.ingest_editable_files:
            if str(p).endswith(".py"):
                return FileSet._norm(p)
        # Fall back to the contract default if mis-configured.
        return "graphsearch/src/search/search.py"

    def _ingest_hash(self, file_set) -> str:
        """12-hex content hash over ONLY the ingestion ``.ts`` overlay file(s),
        sorted by relpath (order-independent). Search-only edits -> same hash ->
        cache hit -> no re-ingest."""
        h = hashlib.sha256()
        for rel in sorted(self._ingest_files()):
            content = file_set.overlay.get(rel, "")
            h.update(rel.encode())
            h.update(b"\0")
            h.update(content.encode())
            h.update(b"\0")
        return h.hexdigest()[:12]

    # ================================================================== #
    # (A)+(B) build/cache the graph (serialized single-flight)            #
    # ================================================================== #
    def _ensure_graph(self, file_set, ingest_hash, graphdb):
        """Ensure the graph for ``ingest_hash`` is the one loaded in the shared
        single DB. Cache hit (loaded_hash == ingest_hash) -> skip ingest, return
        the cached cost. Else SERIALIZE (in-process lock + cross-process file
        lock) and re-ingest: wipe + ``tsx ingest.ts`` (ingest.ts itself drops &
        recreates the constraints + ``corpus_ft`` index per the CONTRACT).

        Returns ``(error_or_None, cost{tokens, usd})``. ``error`` is a non-empty
        string (the tsx stderr / failure reason) when ingestion failed.

        FUTURE (Neo4j Enterprise multi-db isolation): instead of wipe-and-rebuild
        of the default DB keyed by a loaded-hash marker, create a DISTINCT database
        named ``graphdb`` per candidate and pass ``NEO4J_DB=graphdb`` -- then
        concurrent candidates never contend and the cache is just "does the db
        exist". The single-flight lock below becomes unnecessary. Hook left here.
        """
        # Fast path: in-process cache hit (no lock needed for a pure read of an
        # int-sized attribute; the cost re-read is cheap + idempotent).
        if self._loaded_hash == ingest_hash:
            return None, self._read_cost(ingest_hash)

        with _INGEST_LOCK:                       # in-process single-flight
            file_lock = _FileLock(self._lock_path)
            with file_lock:                      # cross-process single-flight
                # Re-check under the lock: another thread/candidate may have just
                # built this exact hash while we waited.
                if self._loaded_hash == ingest_hash or \
                        self._disk_loaded_hash() == ingest_hash:
                    self._loaded_hash = ingest_hash
                    return None, self._read_cost(ingest_hash)

                err = self._run_ingest(file_set, ingest_hash, graphdb)
                if err is not None:
                    # leave the loaded marker UNSET for this hash: a failed build
                    # must not be cached as loaded.
                    return err, self._read_cost(ingest_hash)

                self._loaded_hash = ingest_hash
                self._write_disk_loaded_hash(ingest_hash)
                return None, self._read_cost(ingest_hash)

    def _run_ingest(self, file_set, ingest_hash, graphdb):
        """Materialize the ``extract.ts`` overlay into the graphmod tree, run
        ``npx tsx ingest.ts <corpus> --db <graphdb> --hash <hash>`` (cwd=graphmod),
        and restore the tree afterwards. -> None on success, else an error string
        (timeout / nonzero exit / tsx build error -- carries stderr for repair).

        The .ts overlay is written IN PLACE into ``graphmod_dir_abs`` (so
        ``cwd=graphmod_dir_abs`` resolves node_modules + the relative ingest.ts
        path), then restored in a ``finally`` so the worktree stays pristine. This
        is safe because the whole block runs under the ingest single-flight lock.
        """
        cfg = self._cfg
        backups: dict = {}                       # abspath -> original bytes|None
        try:
            # write each ingestion .ts overlay file into its repo path
            for rel in self._ingest_files():
                abspath = os.path.join(cfg.repo_root, rel)
                backups[abspath] = (open(abspath, "rb").read()
                                    if os.path.exists(abspath) else None)
                os.makedirs(os.path.dirname(abspath), exist_ok=True)
                with open(abspath, "w") as f:
                    f.write(file_set.overlay.get(rel, ""))

            env = dict(os.environ)
            env.update({
                "NEO4J_URI": cfg.neo4j_url,
                "NEO4J_USER": cfg.neo4j_user,
                "NEO4J_PASS": cfg.neo4j_password,
                # default DB on Community single-db; the marker keys the cache.
                "NEO4J_DB": getattr(cfg, "neo4j_database", "neo4j") or "neo4j",
            })
            # ingest.ts relpath under the graphmod cwd (CONTRACT CLI).
            ingest_ts = os.path.join("src", "ingestion", "ingest.ts")
            # --wipe: candidate isolation on the shared single DB (each build starts
            # from an empty graph, so a prior candidate's nodes never leak in).
            cmd = ["npx", "tsx", ingest_ts, cfg.corpus_dir_abs,
                   "--db", graphdb, "--hash", ingest_hash, "--wipe"]
            try:
                proc = subprocess.run(
                    cmd, cwd=cfg.graphmod_dir_abs, env=env,
                    capture_output=True, text=True,
                    timeout=self._ingest_timeout_s)
            except subprocess.TimeoutExpired as e:
                return (f"ingest timeout after {self._ingest_timeout_s}s: "
                        f"{(e.stderr or '')[:1500]}")
            except FileNotFoundError as e:
                # npx/tsx not installed -> a build-environment error, surfaced for
                # the operator (NOT a candidate bug). Still rejects the candidate.
                return f"ingest tooling missing ({e}); is `npx tsx` installed?"

            if proc.returncode != 0:
                # tsx type/build error or schema failure -> reject + self-repair.
                tail = (proc.stderr or proc.stdout or "")[:2000]
                return f"ingest exited {proc.returncode}: {tail}"

            # success: optionally validate the IngestReport JSON on stdout.
            self._last_ingest_report = self._parse_report(proc.stdout)
            return None
        finally:
            # restore the worktree files exactly as they were.
            for abspath, original in backups.items():
                try:
                    if original is None:
                        if os.path.exists(abspath):
                            os.remove(abspath)
                    else:
                        with open(abspath, "wb") as f:
                            f.write(original)
                except OSError:
                    pass

    @staticmethod
    def _parse_report(stdout: str) -> dict:
        """Best-effort parse of the IngestReport JSON emitted on stdout
        (``{nodes, rels, byLabel, costPath}``). Non-fatal: a missing/garbled
        report just yields ``{}`` (ingestion already succeeded by exit code)."""
        for line in reversed((stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {}

    # ================================================================== #
    # ingest_cost.json  (TS writes, we read)                              #
    # ================================================================== #
    def _read_cost(self, ingest_hash) -> dict:
        """Read ``{tokens, usd}`` from ``ingest_cost.json`` (CONTRACT path). Prefer
        the per-hash file ``graphmod/.ingest_cost/<hash>.json`` when present; fall
        back to the stable last-run ``ingest_cost_path_abs``. Defaults to zeros
        (a zero-LLM seed ingest writes ``{tokens:0, usd:0.0}``)."""
        candidates = [
            os.path.join(self._cfg.graphmod_dir_abs, ".ingest_cost",
                         f"{ingest_hash}.json"),
            self._cfg.ingest_cost_path_abs,
        ]
        for path in candidates:
            try:
                with open(path) as f:
                    d = json.load(f)
                return {"tokens": int(d.get("tokens", 0) or 0),
                        "usd": float(d.get("usd", 0.0) or 0.0)}
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                continue
        return {"tokens": 0, "usd": 0.0}

    # ================================================================== #
    # loaded-hash marker (cross-restart cache key)                        #
    # ================================================================== #
    def _disk_loaded_hash(self) -> Optional[str]:
        try:
            with open(self._marker_path) as f:
                return json.load(f).get("hash")
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _write_disk_loaded_hash(self, ingest_hash) -> None:
        try:
            os.makedirs(os.path.dirname(self._marker_path), exist_ok=True)
            tmp = self._marker_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"hash": ingest_hash, "ts": time.time()}, f)
            os.replace(tmp, self._marker_path)
        except OSError:
            pass

    # ================================================================== #
    # (D) crashed vector / rows                                           #
    # ================================================================== #
    def _crashed_vector(self, n_queries, ingest_cost) -> MetricVector:
        """A zeroed-quality, crashed MetricVector (the ingestion edit is rejected).
        ingest cost is still recorded so a doomed-but-expensive build is visible."""
        mv = MetricVector(
            quality={k: 0.0 for k in MCQ_QUALITY_KEYS},
            crashed=True,
            crashed_frac=1.0,
            primary_key=MCQ_PRIMARY,
            ingest_usd=float(ingest_cost.get("usd", 0.0)),
            ingest_tokens=int(ingest_cost.get("tokens", 0)),
            n_queries=int(n_queries),
        )
        mv.sanitize()
        return mv

    def _crashed_rows(self, idxs, ingest_err) -> list:
        """Reflect-shaped rows for a crashed ingest -- every query carries the tsx
        stderr in ``error`` so the mutator self-repairs the bad ingestion edit
        (``fast_loop._failure_record`` reads ``r.get("error")``)."""
        rows = []
        for idx in idxs:
            q_text, q_id, _gold = self._sub.example(idx)
            rows.append({
                "idx": int(idx), "q_id": q_id, "query": q_text,
                "answer_ids": [],
                "metrics": {"recall@20": 0.0, "hit@1": 0.0, "hit@5": 0.0,
                            "mrr": 0.0},
                "retrieved": [], "gold_ranks": {}, "recall@100": 0.0,
                "error": ingest_err,
                "graph_attribution": "NOT_INGESTED (ingestion failed to build "
                                     "the graph; fix the extract.ts edit)",
                "chosen_idx": None, "gold_idx": None,
                "openbook_correct": False, "closedbook_correct": False,
                "retrieval_hit": False, "context_preview": "",
            })
        return rows

    # ================================================================== #
    # (E) attribution                                                     #
    # ================================================================== #
    def _attribute(self, rows, idxs, graphdb, target) -> None:
        """For each MISSED row (retrieval_hit == 0), probe the built graph
        read-only and classify the miss. Mutates ``rows`` in place, setting
        ``row["graph_attribution"]``. Builds the probe graph LAZILY -- only when
        there is at least one miss -- and tolerates a probe failure (leaves the
        recall-based attribution in place)."""
        missed = [(i, idx, r) for i, (idx, r) in enumerate(zip(idxs, rows))
                  if not r.get("retrieval_hit")]
        if not missed:
            return

        inspector = self._build_inspector(graphdb)
        if inspector is None:
            return

        for _i, idx, row in missed:
            try:
                _q, _q_id, gold = self._sub.example(idx)
                src = gold.get("source") or ""
                # multi-gold (list) -> probe the first gold id (diagnostic only).
                gold_id = (src[0] if isinstance(src, (list, tuple)) and src else src)
                if not gold_id or not isinstance(gold_id, str):
                    continue  # no gold node id -> nothing to probe
                diag = (target.diagnostics or {}).get(row.get("query"), {})
                retrieved_ids = diag.get("retrieved_ids")
                row["graph_attribution"] = inspector.attribute_miss(
                    gold_id, seed_ids=None, retrieved_ids=retrieved_ids)
            except Exception:  # noqa: BLE001 - a probe failure must never crash scoring
                continue

    def _build_inspector(self, graphdb):
        """Construct a read-only GraphInspector over a Neo4j graph scoped to
        ``graphdb``. Returns ``None`` (probe disabled) if langchain_neo4j is
        unavailable or the connection fails -- attribution is a diagnostic, never
        load-bearing for the score."""
        try:
            from langchain_neo4j import Neo4jGraph  # deferred dep + DB
            cfg = self._cfg
            graph = Neo4jGraph(
                url=cfg.neo4j_url, username=cfg.neo4j_user,
                password=cfg.neo4j_password, database=graphdb)
            return GraphInspector(graph, db=graphdb)
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------- #
# Cross-process single-flight file lock (POSIX flock)                          #
# --------------------------------------------------------------------------- #
class _FileLock:
    """A blocking advisory file lock so candidates in DIFFERENT processes
    (``num_workers > 1``) serialize their ingest+swap against the shared single
    Neo4j DB. POSIX ``fcntl.flock``; degrades to a no-op where unavailable (the
    in-process ``_INGEST_LOCK`` still serializes within one process)."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fd = None

    def __enter__(self):
        try:
            import fcntl
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            self._fd = open(self._path, "w")
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001 - lock is best-effort cross-process
            self._fd = None
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            try:
                import fcntl
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    self._fd.close()
                except OSError:
                    pass
                self._fd = None
        return False
