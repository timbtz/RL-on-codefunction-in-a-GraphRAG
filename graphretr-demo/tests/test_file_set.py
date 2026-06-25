"""Unit tests for the multi-file artifact (artifact/file_set.py): overlay/sha,
edit_distance, diff, materialize round-trip, and persistence. No infra needed.
"""
import os
import tempfile

from graphretr_opt.artifact.file_set import FileSet


def _base(tmp):
    """A tiny 2-file base checkout under tmp/base."""
    base = os.path.join(tmp, "base")
    os.makedirs(os.path.join(base, "pkg"))
    with open(os.path.join(base, "pkg", "a.py"), "w") as f:
        f.write("X = 1\n")
    with open(os.path.join(base, "pkg", "b.py"), "w") as f:
        f.write("Y = 2\n")
    return base


def test_from_base_and_sha():
    with tempfile.TemporaryDirectory() as tmp:
        base = _base(tmp)
        fs = FileSet.from_base(base, ("pkg/a.py",))
        assert set(fs.overlay) == {"pkg/a.py"}
        assert fs.overlay["pkg/a.py"] == "X = 1\n"
        # sha is stable and overlay-content-keyed (base_dir excluded)
        fs2 = FileSet(base + "_elsewhere", {"pkg/a.py": "X = 1\n"}, ("pkg/a.py",))
        assert fs.sha == fs2.sha


def test_with_overlay_and_edit_distance():
    fs = FileSet("/b", {"pkg/a.py": "X = 1\n"}, ("pkg/a.py",))
    fs2 = fs.with_overlay({"pkg/a.py": "X = 2\n"})
    assert fs2.overlay["pkg/a.py"] == "X = 2\n"
    assert fs.sha != fs2.sha
    assert fs.edit_distance(fs2) >= 1
    assert fs.edit_distance(fs) == 0


def test_materialize_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        base = _base(tmp)
        fs = FileSet.from_base(base, ("pkg/a.py",)).with_overlay(
            {"pkg/a.py": "X = 99\n"})
        dest = os.path.join(tmp, "dest")
        fs.materialize(dest)
        # overlay file overwritten, untouched file copied verbatim
        assert open(os.path.join(dest, "pkg", "a.py")).read() == "X = 99\n"
        assert open(os.path.join(dest, "pkg", "b.py")).read() == "Y = 2\n"


def test_save_py_path_writes_primary():
    with tempfile.TemporaryDirectory() as tmp:
        fs = FileSet("/b", {"pkg/a.py": "X = 1\n"}, ("pkg/a.py",))
        out = os.path.join(tmp, "best.py")
        fs.save(out)
        assert open(out).read() == "X = 1\n"


def test_to_from_dict_roundtrip():
    fs = FileSet("/b", {"pkg/a.py": "X = 1\n"}, ("pkg/a.py",))
    fs2 = FileSet.from_dict(fs.to_dict())
    assert fs2.sha == fs.sha
    assert fs2.editable == fs.editable
    assert fs2.base_dir == fs.base_dir
