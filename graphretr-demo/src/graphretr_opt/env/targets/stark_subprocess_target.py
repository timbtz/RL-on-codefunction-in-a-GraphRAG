"""StarkSubprocessTarget -- the STaRK SearchTarget adapter.

The STaRK analogue of SubprocessSearchTarget. For each candidate FileSet it
materializes a throwaway copy of `starksearch/src` with the overlay applied,
spawns `graphretr_opt.env.targets._worker_stark`, enforces a hard wall-clock
kill, and parses each query's `{node_id: score}` pred + cost. Same isolation
model as the graphsearch target (process boundary + the throwaway dir + the
kill); the difference is the STaRK result is a pred dict, not a retrieved-context
string, so it does NOT use SearchResult/CostMeter -- it returns a plain
`{query: {pred, cost, error}}` map the STaRK reward
(`starksearch.reward.StarkRewardAdapter`) scores via the STaRK Evaluator.

The candidate is a whole `FileSet` over `starksearch/src` (the editable service
file overlaid on the immutable base). The server-side caps that protect the
shared FalkorDB container live in the harness firewall
(`stark_harness.qa_runner`), not here.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

_WORKER_MODULE = "graphretr_opt.env.targets._worker_stark"


class StarkSubprocessTarget:
    def __init__(self, falkor_cfg, opt_src_dir, repo_root, cfg_overrides=None,
                 python_exe=None, tmp_root=None, query_concurrency=1):
        """falkor_cfg: {host, port, graph_name} for the worker's FalkorDB backend.
        opt_src_dir: graphretr-demo `src` (worker PYTHONPATH so `-m graphretr_opt`
            resolves).
        repo_root: monorepo root (worker PYTHONPATH so `starksearch` resolves).
        cfg_overrides: kwargs the worker passes to load_config() to rebuild the
            harness clients with the SAME embedder/models/caches as the parent
            (pass root so runs_dir caches are shared).
        tmp_root: parent dir for throwaway overlays (default: system temp).
        query_concurrency: how many of this batch's queries the worker runs in
            parallel (each on its own service+sink). 1 = sequential.
        """
        self._falkor = falkor_cfg
        self._opt_src = opt_src_dir
        self._repo_root = repo_root
        self._cfg_overrides = cfg_overrides or {}
        self._python = python_exe or sys.executable
        self._tmp_root = tmp_root
        self._query_concurrency = max(1, int(query_concurrency or 1))

    def run(self, file_set, queries, timeout_s: float) -> dict:
        """file_set: the candidate FileSet over starksearch/src.
        -> {query: {pred, cost, error}}."""
        queries = list(queries)
        tmp = tempfile.mkdtemp(prefix="stark_cand_", dir=self._tmp_root)
        overlay_dir = os.path.join(tmp, "src")
        try:
            file_set.materialize(overlay_dir)
            job = {
                "overlay_dir": overlay_dir,
                "queries": queries,
                "falkor_cfg": self._falkor,
                "cfg_overrides": self._cfg_overrides,
                "query_concurrency": self._query_concurrency,
            }
            return self._spawn(job, queries, timeout_s)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _spawn(self, job, queries, timeout_s):
        env = dict(os.environ)
        # Worker needs graphretr_opt importable to run `-m`, and starksearch (the
        # base backend/embedder/config the harness imports) importable too; the
        # candidate's edited service overlay is added to sys.path inside the
        # worker itself.
        env["PYTHONPATH"] = os.pathsep.join(
            [self._opt_src, self._repo_root, env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        proc = subprocess.Popen(
            [self._python, "-m", _WORKER_MODULE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, start_new_session=True)
        try:
            out, err = proc.communicate(input=json.dumps(job), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._kill(proc)
            return self._all_error(queries, f"subprocess timed out (> {timeout_s}s)")
        if proc.returncode != 0 and not out:
            return self._all_error(
                queries, f"worker exited {proc.returncode}: {err[:500]}")
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            return self._all_error(
                queries, f"worker emitted non-JSON: {out[:300]} / {err[:300]}")
        if "error" in payload:
            return self._all_error(queries, f"worker fatal: {payload['error']}")
        return self._parse(payload.get("results", {}), queries)

    @staticmethod
    def _kill(proc):
        """Kill the whole process group on overrun (the worker may spawn children)."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    @staticmethod
    def _parse(results, queries):
        out = {}
        for q in queries:
            r = results.get(q)
            if r is None:
                out[q] = {"pred": {}, "cost": {}, "error": "no result returned for query"}
                continue
            pred = {int(k): float(v) for k, v in (r.get("pred") or {}).items()}
            out[q] = {"pred": pred, "cost": r.get("cost") or {}, "error": r.get("error")}
        return out

    @staticmethod
    def _all_error(queries, msg):
        return {q: {"pred": {}, "cost": {}, "error": msg} for q in queries}
