"""StarkSubprocessTarget -- the STaRK SearchTarget adapter.

The STaRK analogue of SubprocessSearchTarget. Spawns
`graphretr_opt.env.targets._worker_stark` with the candidate program source +
FalkorDB/cfg, enforces a hard wall-clock kill, and parses each query's
`{node_id: score}` pred + cost. Same isolation model as the graphsearch target
(process boundary + the throwaway interpreter + the kill); the difference is the
STaRK result is a pred dict, not a retrieved-context string, so it does NOT use
SearchResult/CostMeter -- it returns a plain `{query: {pred, cost, error}}` map
the STaRK reward (`starksearch.reward.StarkRewardAdapter`) scores via the STaRK
Evaluator.

The candidate is a single-file `search(q, G)` SOURCE string (no FileSet overlay
to materialize on this path); the worker exec's it with real builtins (the
loosening of the in-process AST gate). The server-side caps that protect the
shared FalkorDB container live in `starksearch.primitives`, not here.
"""
import json
import os
import signal
import subprocess
import sys
import time

_WORKER_MODULE = "graphretr_opt.env.targets._worker_stark"


class StarkSubprocessTarget:
    def __init__(self, falkor_cfg, opt_src_dir, repo_root, cfg_overrides=None,
                 python_exe=None):
        """falkor_cfg: {host, port, graph_name} for the worker's FalkorDB backend.
        opt_src_dir: graphretr-demo `src` (worker PYTHONPATH so `-m graphretr_opt`
            resolves).
        repo_root: monorepo root (worker PYTHONPATH so `starksearch` resolves).
        cfg_overrides: kwargs the worker passes to load_config() to rebuild G with
            the SAME embedder/models/caches as the parent (pass root so runs_dir
            caches are shared).
        """
        self._falkor = falkor_cfg
        self._opt_src = opt_src_dir
        self._repo_root = repo_root
        self._cfg_overrides = cfg_overrides or {}
        self._python = python_exe or sys.executable

    def run(self, src, queries, timeout_s: float) -> dict:
        """src: the candidate program source. -> {query: {pred, cost, error}}."""
        queries = list(queries)
        job = {
            "candidate_src": src,
            "queries": queries,
            "falkor_cfg": self._falkor,
            "cfg_overrides": self._cfg_overrides,
            "opt_src": self._opt_src,
            "repo_root": self._repo_root,
        }
        return self._spawn(job, queries, timeout_s)

    def _spawn(self, job, queries, timeout_s):
        env = dict(os.environ)
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
