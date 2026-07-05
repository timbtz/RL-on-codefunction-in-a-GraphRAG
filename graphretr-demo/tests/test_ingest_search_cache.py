"""Unit tests for the IngestSearchRewardAdapter single-DB cache logic
(reward/ingest_search.py): the ingest hash (.ts-only inputs + the
``ingest_llm_extraction`` flag), the on-disk loaded-hash marker, the tsx
command shape (``--llm`` lands; ``--db`` never does -- Neo4j Community cannot
CREATE DATABASE), and the shared/exclusive file-lock semantics that keep a
concurrent wipe out from under a running search.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_ingest_search_cache
No Neo4j / tsx / network needed.
"""
import os
import tempfile
import types

from graphretr_opt.artifact.file_set import FileSet
from graphretr_opt.reward import ingest_search as isr

TS_REL = "graphmod/src/ingestion/extract.ts"
PY_REL = "graphsearch/src/search/search.py"


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _cfg(tmp, llm=False):
    return types.SimpleNamespace(
        ingest_editable_files=(TS_REL, PY_REL),
        ingest_llm_extraction=llm,
        root=tmp,  # getattr(cfg, "runs_dir", cfg.root/...) evaluates its default
        runs_dir=os.path.join(tmp, "runs"),
        repo_root=tmp,
        corpus_dir_abs=os.path.join(tmp, "corpus"),
        graphmod_dir_abs=os.path.join(tmp, "graphmod"),
        ingest_cost_path_abs=os.path.join(tmp, "ingest_cost.json"),
        neo4j_url="bolt://localhost:9999", neo4j_user="neo4j",
        neo4j_password="x", neo4j_database="neo4j",
    )


def _adapter(tmp, llm=False):
    return isr.IngestSearchRewardAdapter(None, None, None, _cfg(tmp, llm))


def _fs(ts="// seed ts", py="# seed py"):
    return FileSet("/nonexistent-base", {TS_REL: ts, PY_REL: py},
                   (TS_REL, PY_REL))


def test_search_only_edit_keeps_hash():
    with tempfile.TemporaryDirectory() as tmp:
        a = _adapter(tmp)
        _check("search-only edit -> same ingest hash (cache hit)",
               a._ingest_hash(_fs()) == a._ingest_hash(_fs(py="# edited py")))


def test_ts_edit_changes_hash():
    with tempfile.TemporaryDirectory() as tmp:
        a = _adapter(tmp)
        _check(".ts edit -> new ingest hash (rebuild)",
               a._ingest_hash(_fs()) != a._ingest_hash(_fs(ts="// edited ts")))


def test_llm_flag_changes_hash():
    with tempfile.TemporaryDirectory() as tmp:
        off, on = _adapter(tmp, llm=False), _adapter(tmp, llm=True)
        _check("ingest_llm_extraction toggle -> new ingest hash",
               off._ingest_hash(_fs()) != on._ingest_hash(_fs()))


def test_marker_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        a = _adapter(tmp)
        _check("missing marker reads as None", a._disk_loaded_hash() is None)
        a._write_disk_loaded_hash("abc123def456")
        _check("marker roundtrips", a._disk_loaded_hash() == "abc123def456")


def test_cmd_has_llm_and_never_db():
    with tempfile.TemporaryDirectory() as tmp:
        a = _adapter(tmp, llm=True)
        fs = _fs()
        captured = {}

        class _Proc:
            returncode, stdout, stderr = 0, "{}", ""

        real = isr.subprocess.run
        isr.subprocess.run = lambda cmd, **kw: (captured.update(cmd=cmd),
                                                _Proc())[1]
        try:
            err = a._run_ingest(fs, a._ingest_hash(fs))
        finally:
            isr.subprocess.run = real
        _check("fake ingest succeeds", err is None)
        _check("--llm lands in the tsx command", "--llm" in captured["cmd"])
        _check("--db never passed (Community single-db)",
               "--db" not in captured["cmd"])
        _check("worktree restored after ingest (overlay file removed)",
               not os.path.exists(os.path.join(tmp, TS_REL)))


def test_shared_lock_excludes_exclusive():
    import fcntl
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ingest.lock")
        shared = isr._FileLock(path, shared=True).acquire()
        fd = open(path, "w")
        try:
            blocked = False
            try:  # a wipe (EX) must wait while a search holds the lock SHARED
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                blocked = True
            _check("EX blocks while SH held (no mid-search wipe)", blocked)
            shared.release()
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _check("EX acquires once SH released", True)
        finally:
            fd.close()


def main():
    test_search_only_edit_keeps_hash()
    test_ts_edit_changes_hash()
    test_llm_flag_changes_hash()
    test_marker_roundtrip()
    test_cmd_has_llm_and_never_db()
    test_shared_lock_excludes_exclusive()
    print("\nall ingest_search cache tests passed")


if __name__ == "__main__":
    main()
