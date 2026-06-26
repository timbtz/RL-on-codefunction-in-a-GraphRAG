"""qa_runner -- build + run the editable STaRK search service for one query.

The STaRK analogue of `graphsearch/.../qa_eval/qa_runner.py`. One deep pair:
  build_service(falkor_cfg, cfg_overrides, instrument) -> StarkGraphSearchService
  run_query(service, query) -> dict[int, float]

The service takes fully-constructed clients (a read-only FalkorDB proxy, an
embedder, and a metered JSON-mode LLM client), so we build them directly here --
no DSL, no `RetrievalGraph`. When an `instrument` (CostSink) is supplied the db /
llm clients meter into it, so the optimizer reads db_queries / llm_calls /
rerank_items WITHOUT editing the service: the metering lives in the harness, the
artifact stays clean (mirrors the graphsearch CostSink/proxy pattern).

This module also owns the FIREWALL (`_assert_safe`): the candidate now writes raw
Cypher through `db.cypher`, so the SPpaths-maxLen<=3 wall and the write rejection
that protect the SHARED FalkorDB container live HERE, on the one chokepoint every
candidate query passes through -- NOT in a service method the candidate can edit
or bypass.

Hard rules (eval hygiene + container safety):
  * read-only: writes (CREATE/MERGE/SET/DELETE/REMOVE/DROP, write procs) rejected.
  * `algo.SPpaths` capped at maxLen<=3 (>=4 runs away and ignores the timeout).
  * per-query timeout + a row cap bound any single query.
  * creds / models come from cfg (env-sourced); never hardcoded here.
"""
import json
import os
import re
import time
from dataclasses import dataclass

from starksearch.backends.falkordb import FalkorDBBackend
from starksearch.embedder import make_embedder
from graphretr_opt.config import load_config
from graphretr_opt.env.openai_client import OpenAIBudget

_MAX_QUERY_ROWS = 5000   # hard row cap on a raw db.cypher() result (runaway guard)


@dataclass
class CostSink:
    """Mutable cost accumulator the metered clients write into. One per
    build_service() call; the worker reads it after each query and resets. Same
    axes the in-process RetrievalGraph exposed via getattr deltas (db_queries /
    llm_calls / rerank_items), so the MetricVector the reward builds is
    unchanged."""
    llm_calls: int = 0
    db_queries: int = 0
    rerank_items: int = 0          # items sent to the LLM as context (cost driver)
    tokens_in: int = 0
    tokens_out: int = 0
    usd_cost: float = 0.0

    def reset(self):
        self.llm_calls = self.db_queries = self.rerank_items = 0
        self.tokens_in = self.tokens_out = 0
        self.usd_cost = 0.0

    def snapshot(self) -> dict:
        return {"llm_calls": self.llm_calls, "db_queries": self.db_queries,
                "rerank_items": self.rerank_items, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "usd_cost": self.usd_cost}


# ------------------------------------------------------------------- firewall

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(r"//[^\n]*")
# Write clauses + write procedures. Word-boundary so RETURN/queryNodes/CONTAINS
# (which embed none of these as whole words) pass. `set`/`remove`/`drop` etc. are
# never substrings of the read primitives the seed emits.
_WRITE_RE = re.compile(r"\b(create|merge|set|delete|remove|drop|detach)\b")
_WRITE_PROC_RE = re.compile(
    r"call\s+(db\.create|apoc\.\w*(create|merge|set|delete|add|remove|update))")
_MAXLEN_RE = re.compile(r"maxlen\s*:?\s*(\d+)")


def _assert_safe(query):
    """THE FIREWALL. Reject a candidate Cypher string that would write to or peg
    the SHARED FalkorDB. Raise ValueError (the worker scores it as a per-query
    miss). This is the ONLY enforcement of the SPpaths/write wall now that the
    candidate emits raw Cypher (the old wall lived in RetrievalGraph methods the
    candidate routed around)."""
    raw = str(query)
    q = _COMMENT_LINE.sub(" ", _COMMENT_BLOCK.sub(" ", raw)).lower()
    if _WRITE_RE.search(q) or _WRITE_PROC_RE.search(q):
        raise ValueError(f"unsafe cypher (write clause/proc): {raw[:200]}")
    if "sppaths" in q:
        m = _MAXLEN_RE.search(q)
        if not m or int(m.group(1)) > 3:
            raise ValueError(
                f"unsafe cypher (algo.SPpaths needs maxLen<=3): {raw[:200]}")
    return raw


# ----------------------------------------------------------------- db client

class ReadOnlyGraphClient:
    """Metered + firewalled read-only proxy over FalkorDBBackend -- the `db` the
    service is constructed with. `cypher()` is the one chokepoint every candidate
    query passes through: firewall -> meter -> capped read-only query. The
    label / rel_type / ntype allowlists are read off the live graph once at
    construction (exactly as the old RetrievalGraph did)."""

    def __init__(self, backend, sink: CostSink, cfg):
        self._backend = backend
        self._sink = sink
        self._cfg = cfg
        labels = sorted(r[0] for r in self._ro(
            "CALL db.labels() YIELD label RETURN label"))
        rels = sorted(r[0] for r in self._ro(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType"))
        ntypes = sorted(r[0] for r in self._ro(
            "MATCH (n:Entity) RETURN DISTINCT n.ntype"))
        # vector-indexed per-type labels (drop the umbrella 'Entity' label)
        self._labels = tuple(l for l in labels if l != "Entity")
        self._rel_types = tuple(rels)
        self._ntypes = tuple(ntypes)

    def _ro(self, query, params=None):
        """Internal trusted read (allowlist bootstrap): no firewall/metering."""
        return self._backend.ro_query(
            query, params=params or {}, timeout_ms=self._cfg.query_timeout_ms)

    def cypher(self, query, params=None) -> list:
        """Run ONE read-only Cypher query. -> list of rows (each a plain list),
        truncated to the row cap. Writes / runaway SPpaths are rejected by the
        firewall BEFORE the db_queries meter increments (a rejected query never
        reaches the backend and is not counted)."""
        _assert_safe(query)
        params = params if isinstance(params, dict) else {}
        self._sink.db_queries += 1
        rows = self._backend.ro_query(
            query, params=params, timeout_ms=self._cfg.query_timeout_ms)
        return [list(r) for r in rows][:_MAX_QUERY_ROWS]

    def get_text(self, ids, limit=50) -> dict:
        """Node documents for the given ids. -> {node_id: text}. Used by the LLM
        client to assemble context_ids; the service ports its own _get_text on
        top of cypher()."""
        ids = [int(i) for i in ids][:int(limit)]
        if not ids:
            return {}
        rows = self.cypher(
            "UNWIND $ids AS i MATCH (n:Entity {id:i}) RETURN n.id, n.text",
            {"ids": ids})
        return {int(r[0]): str(r[1]) for r in rows}

    @property
    def labels(self):
        return self._labels

    @property
    def rel_types(self):
        return self._rel_types

    @property
    def ntypes(self):
        return self._ntypes


# --------------------------------------------------------------- embedder

class Embedder:
    """Thin adapter exposing `encode(text) -> list[float]` over the configured
    QueryEmbedder / OpenAIQueryEmbedder (which return np.ndarray). The service
    passes the result straight into a `vecf32($vec)` Cypher param, so a plain
    list is what FalkorDB needs."""

    def __init__(self, inner):
        self._inner = inner

    def encode(self, text) -> list:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedder.encode needs a non-empty string")
        v = self._inner.encode(text)
        return v.tolist() if hasattr(v, "tolist") else list(v)


# --------------------------------------------------------------- llm client

class LlmClient:
    """One metered, disk-cached, JSON-mode LLM call -- the raw capability the
    service's extract/rerank/reformulate helpers are built on. Ports
    RetrievalGraph.llm verbatim (budget ceiling + cache by (model, system,
    full_user)); the per-operator caches are NOT reproduced -- this one
    (model,system,user) cache at the single chokepoint subsumes them. Counts the
    call on `sink.llm_calls` and the supplied context on `sink.rerank_items`."""

    def __init__(self, budget, cfg, sink: CostSink, runs_dir, db=None):
        self._budget = budget
        self._cfg = cfg
        self._sink = sink
        self._db = db
        self._disk = os.path.join(runs_dir, "llm_cache.json")
        self._mem = None

    def __call__(self, system, user, context_ids=None, model=None,
                 max_tokens=600) -> dict:
        if not (isinstance(system, str) and system.strip()):
            raise ValueError("system must be a non-empty string")
        if not (isinstance(user, str) and user.strip()):
            raise ValueError("user must be a non-empty string")
        if self._budget is None:
            raise ValueError("llm() is not enabled (no OPENAI_API_KEY)")
        model = model or self._cfg.rerank_model
        ctx_ids, seen = [], set()
        if context_ids is not None:
            for nid in (int(x) for x in context_ids):
                if nid not in seen and len(ctx_ids) < self._cfg.rerank_pool_max:
                    seen.add(nid)
                    ctx_ids.append(nid)
        # meters: charge the call + the context the program asked for.
        self._sink.llm_calls += 1
        self._sink.rerank_items += len(ctx_ids)
        system, user = system[:4000], user[:6000]   # guard program-authored prompts
        full_user = user
        if ctx_ids and self._db is not None:
            texts = self._db.get_text(ctx_ids, limit=len(ctx_ids))
            lines = "\n".join(f"[{nid}] {texts.get(nid, '')[:500]}" for nid in ctx_ids)
            full_user = f"{user}\n\nNodes:\n{lines}"
        if self._mem is None:
            self._mem = self._load_disk()
        ckey = "\x00".join([model, system, full_user])
        hit = self._mem.get(ckey)
        if hit is not None:
            return json.loads(json.dumps(hit))
        raw = self._budget.chat_json(system, full_user, model=model,
                                     max_tokens=max_tokens)
        raw = raw if isinstance(raw, dict) else {}
        self._mem[ckey] = raw
        os.makedirs(os.path.dirname(self._disk), exist_ok=True)
        # Atomic write: dump to a temp file then rename, so a kill mid-write can
        # never truncate the cache into the corrupt-JSON state that crashes the
        # next run on load (that exact failure mode produced runs/llm_cache.json
        # corruption after run10c was killed).
        tmp = f"{self._disk}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self._mem, f)
        os.replace(tmp, self._disk)
        return json.loads(json.dumps(raw))

    def _load_disk(self) -> dict:
        """Load the disk cache, tolerating a corrupt/truncated file. A killed run
        can leave a half-written cache; rather than crash every subsequent run on
        the JSONDecodeError, we move the bad file aside (.corrupt) and start fresh
        -- the cache is a perf optimization, never a source of truth."""
        if not os.path.exists(self._disk):
            return {}
        try:
            with open(self._disk) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            bad = f"{self._disk}.corrupt"
            try:
                os.replace(self._disk, bad)
            except OSError:
                bad = "(could not move)"
            print(f"[llm_cache] corrupt cache ignored ({type(e).__name__}); "
                  f"moved to {bad}, starting fresh", flush=True)
            return {}


# --------------------------------------------------------------- build / run

def build_service(falkor_cfg, cfg_overrides=None, instrument: CostSink = None):
    """Construct StarkGraphSearchService with a read-only FalkorDB proxy +
    embedder + metered LLM client, optionally metered into `instrument`.

    falkor_cfg:    {host, port, graph_name}
    cfg_overrides: kwargs forwarded to load_config() so the worker rebuilds the
                   SAME embedder/models/caches as the parent (pass root to share
                   the runs_dir caches).
    instrument:    a CostSink to meter into (None = a throwaway sink).
    """
    cfg = load_config(**(cfg_overrides or {}))
    backend = FalkorDBBackend(
        falkor_cfg["host"], falkor_cfg["port"], falkor_cfg["graph_name"])
    # One-time engine safety config (global to our container alone).
    backend.configure("TIMEOUT_MAX", 10_000)
    backend.configure("QUERY_MEM_CAPACITY", 268_435_456)
    budget = None
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        budget = OpenAIBudget(os.path.join(cfg.runs_dir, "openai_usage.json"),
                              ceiling_usd=cfg.openai_budget_usd)
    sink = instrument if instrument is not None else CostSink()
    db = ReadOnlyGraphClient(backend, sink, cfg)
    embedder = Embedder(make_embedder(cfg, budget))
    llm = LlmClient(budget, cfg, sink, cfg.runs_dir, db=db)
    # Imported lazily from the overlay so the EDITED service file (materialized
    # into the throwaway tree, on sys.path ahead of the base) is the one that
    # loads -- never an absolute base-path import.
    from stark_search.stark_graph_search_service import StarkGraphSearchService
    return StarkGraphSearchService(db, embedder, llm)


def run_query(service, query: str) -> dict:
    """Run one query through the service synchronously. -> dict[int, float]
    (node id -> score). Errors propagate to the caller (the worker serializes
    them as a per-query crash)."""
    start = time.perf_counter()
    try:
        return service.search(query)
    finally:
        time.perf_counter() - start  # latency is metered per-query by the worker
