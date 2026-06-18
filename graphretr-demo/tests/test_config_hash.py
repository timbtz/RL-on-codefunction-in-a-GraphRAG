"""Unit tests for Phase A: the resolved-config snapshot + config_hash.

Same inputs => same hash; an env/CLI override that changes a run-affecting field
=> a different hash (the silent-drift detector); `root` (a machine path) is
excluded so the hash is portable; and the dumped resolved_config.yaml hashes to
exactly the logged config_hash.

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_config_hash
No FalkorDB / no network / no MLflow server needed.
"""
import hashlib
import os
import tempfile
from dataclasses import replace

from graphretr_opt.config import load_config, dump_resolved_config


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def test_hash_deterministic_and_root_independent():
    with tempfile.TemporaryDirectory() as d:
        c1 = load_config(root=d)
        c2 = load_config(root=d)
        _check("identical inputs -> identical hash",
               c1.config_hash() == c2.config_hash())
        _check("hash is 12 hex chars",
               len(c1.config_hash()) == 12 and all(ch in "0123456789abcdef"
                                                   for ch in c1.config_hash()))
        c3 = load_config(root="/some/other/checkout/path")
        _check("root excluded -> hash portable across checkouts",
               c1.config_hash() == c3.config_hash())


def test_env_override_changes_hash():
    """A CLI/env override changing a field under an unchanged git_sha must flip
    the hash -- otherwise the drift is invisible (the gap Phase A closes)."""
    with tempfile.TemporaryDirectory() as d:
        base = load_config(root=d)
        os.environ["MLFLOW_URL"] = "http://drift-env-only:5000"
        try:
            drifted = load_config(root=d)
        finally:
            del os.environ["MLFLOW_URL"]
        _check("env override captured in resolved field",
               drifted.mlflow_url == "http://drift-env-only:5000")
        _check("env override -> different hash",
               base.config_hash() != drifted.config_hash())
        # a kwargs override of any other run-affecting field too
        _check("changed gate_seed -> different hash",
               base.config_hash() != replace(base, gate_seed=base.gate_seed + 1).config_hash())


def test_dump_matches_logged_hash():
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(root=d)
        run_dir = os.path.join(d, "runs", "trk")
        path, h = dump_resolved_config(cfg, run_dir)
        text = open(path).read()
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        _check("dumped yaml hashes to the returned config_hash",
               content_hash == h == cfg.config_hash())
        _check("resolved_config.yaml omits machine-specific root",
               "root:" not in text)
        _check("resolved_config.yaml carries a run-affecting field",
               "gate_seed:" in text)


def main():
    test_hash_deterministic_and_root_independent()
    test_env_override_changes_hash()
    test_dump_matches_logged_hash()
    print("\nall config_hash tests passed")


if __name__ == "__main__":
    main()
