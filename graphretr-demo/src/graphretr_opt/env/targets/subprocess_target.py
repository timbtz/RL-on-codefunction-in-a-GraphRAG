"""SubprocessSearchTarget -- the production SearchTarget adapter.

For each candidate FileSet:
  1. materialize a throwaway copy of graphsearch/src with the overlay applied
     (shutil.copytree -- ~a handful of files, so per-candidate setup is cheap; no
     git worktree needed since it is an in-repo folder, not a separate repo);
  2. spawn `python -m graphretr_opt.env.targets._worker`, feeding it the job JSON
     (overlay_dir + the question strings + Neo4j/LLM cfg) on stdin;
  3. enforce a hard wall-clock timeout (kill the whole process group on overrun
     -- this REPLACES sandbox.py's SIGALRM-in-main-thread);
  4. parse the per-query {context, cost}, aggregate a CostMeter, and ALWAYS
     remove the throwaway dir (try/finally), even on timeout.

Isolation = the process boundary + the throwaway dir + the kill. A candidate
that fails to import, crashes, or hangs is contained and scored as a miss for
its queries; it never takes down the optimizer.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from ..search_target import CostMeter, SearchResult

_WORKER_MODULE = "graphretr_opt.env.targets._worker"


class SubprocessSearchTarget:
    def __init__(self, neo4j_cfg, llm_cfg, opt_src_dir, service_relpath=None,
                 service_kwargs=None, python_exe=None, tmp_root=None,
                 query_concurrency=1, worker_module=None):
        """neo4j_cfg / llm_cfg: passed through to the worker's build_service.
        opt_src_dir: the graphretr-demo `src` dir, put on the worker's PYTHONPATH
            so `python -m graphretr_opt...._worker` resolves.
        service_relpath: which editable target class the worker builds.
        python_exe: interpreter for the worker (default: this process's).
        tmp_root: parent dir for throwaway overlays (default: system temp).
        query_concurrency: how many of this batch's queries the worker runs in
            parallel (each on its own service+sink). 1 = sequential.
        worker_module: the `-m` module the subprocess runs. Defaults to the
            graphsearch worker; the STaRK boot points it at `_worker_stark` so the
            SAME spawn/overlay/wall-clock-kill machinery drives both services.
        """
        self._neo4j = neo4j_cfg
        self._llm = llm_cfg
        self._opt_src = opt_src_dir
        self._service_relpath = service_relpath
        self._service_kwargs = service_kwargs or {}
        self._python = python_exe or sys.executable
        self._tmp_root = tmp_root
        self._query_concurrency = max(1, int(query_concurrency or 1))
        self._worker_module = worker_module or _WORKER_MODULE

    def run(self, file_set, queries, timeout_s: float) -> dict:
        queries = list(queries)
        tmp = tempfile.mkdtemp(prefix="graphsearch_cand_", dir=self._tmp_root)
        overlay_dir = os.path.join(tmp, "src")
        try:
            file_set.materialize(overlay_dir)
            job = {
                "overlay_dir": overlay_dir,
                "queries": queries,
                "neo4j_cfg": self._neo4j,
                "llm_cfg": self._llm,
                "service_relpath": self._service_relpath,
                "service_kwargs": self._service_kwargs,
                "query_concurrency": self._query_concurrency,
            }
            return self._spawn(job, queries, timeout_s)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _spawn(self, job, queries, timeout_s):
        env = dict(os.environ)
        # Worker needs graphretr_opt importable to run `-m`; the candidate's
        # graphsearch overlay is added to sys.path inside the worker itself.
        env["PYTHONPATH"] = os.pathsep.join(
            [self._opt_src, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        t0 = time.time()
        proc = subprocess.Popen(
            [self._python, "-m", self._worker_module],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, start_new_session=True)
        try:
            out, err = proc.communicate(input=json.dumps(job), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._kill(proc)
            out, err = "", f"timeout after {timeout_s}s"
            return self._all_error(queries,
                                   f"subprocess timed out (> {timeout_s}s)",
                                   time.time() - t0)
        elapsed = time.time() - t0
        if proc.returncode != 0 and not out:
            return self._all_error(
                queries, f"worker exited {proc.returncode}: {err[:500]}", elapsed)
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            return self._all_error(
                queries, f"worker emitted non-JSON: {out[:300]} / {err[:300]}",
                elapsed)
        if "error" in payload:
            return self._all_error(
                queries, f"worker fatal: {payload['error']}", elapsed)
        return self._parse(payload.get("results", {}), queries, elapsed)

    @staticmethod
    def _kill(proc):
        """Kill the whole process group (the service spawns a ThreadPoolExecutor;
        children must die with the worker on timeout)."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _parse(self, results, queries, elapsed):
        per_query_lat = elapsed / max(1, len(queries))
        out = {}
        for q in queries:
            r = results.get(q)
            if r is None:
                out[q] = SearchResult(
                    context="", cost=CostMeter(latency_s=per_query_lat),
                    error="no result returned for query")
                continue
            cost = CostMeter.from_dict(r.get("cost", {}))
            cost.latency_s = per_query_lat
            out[q] = SearchResult(context=r.get("context", "") or "",
                                  cost=cost, error=r.get("error"))
        return out

    @staticmethod
    def _all_error(queries, msg, elapsed):
        per_query_lat = elapsed / max(1, len(queries))
        return {q: SearchResult(context="",
                                cost=CostMeter(latency_s=per_query_lat),
                                error=msg) for q in queries}
