"""Unit test for seed-from-champion (archipelago seed-chaining): overlaying a
saved champion's primary file onto a fresh-from-base seed FileSet changes the
seed sha and makes the primary file's content == the champion. Mirrors the
campaign.final_test reload pattern (seed.with_overlay({primary: champ_src})).

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_seed_from_champion
No FalkorDB / no network needed.
"""
import os
import tempfile

from graphretr_opt.artifact.file_set import FileSet


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _base_tree(d):
    """A tiny 2-file editable base checkout under `d`; returns (base_dir, rel)."""
    rel = "svc/search.py"
    os.makedirs(os.path.join(d, "svc"))
    with open(os.path.join(d, rel), "w") as f:
        f.write("def search(q):\n    return []  # BASE\n")
    return d, rel


def test_overlay_changes_sha_and_content():
    with tempfile.TemporaryDirectory() as d:
        base_dir, rel = _base_tree(d)
        seed = FileSet.from_base(base_dir, (rel,), family="stark_search")
        base_sha = seed.sha
        champ_src = "def search(q):\n    return ['CHAMP']  # evolved\n"
        champ_path = os.path.join(d, "champion.py")
        with open(champ_path, "w") as f:
            f.write(champ_src)

        # the exact overlay the campaign._boot_function seed-champion path runs
        champ = open(champ_path, encoding="utf-8").read()
        seeded = seed.with_overlay({seed.primary_file: champ})

        _check("overlay changes the seed sha", seeded.sha != base_sha)
        _check("primary file content == champion source",
               seeded.src_of(seeded.primary_file) == champ_src)
        _check("base, editable, family preserved",
               seeded.base_dir == seed.base_dir
               and seeded.editable == seed.editable
               and seeded.family == seed.family)
        # re-overlaying the SAME champion is idempotent (stable sha)
        again = seed.with_overlay({seed.primary_file: champ})
        _check("re-overlay is deterministic (same sha)", again.sha == seeded.sha)


def _run():
    test_overlay_changes_sha_and_content()
    print("all seed-from-champion tests passed")


if __name__ == "__main__":
    _run()
