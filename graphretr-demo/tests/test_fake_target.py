"""Unit tests for FakeSearchTarget (env/targets/fake_target.py): port
conformance, determinism, canned-context injection.
"""
from graphretr_opt.artifact.file_set import FileSet
from graphretr_opt.env.search_target import CostMeter, SearchResult, SearchTarget
from graphretr_opt.env.targets.fake_target import FakeSearchTarget

FS = FileSet("/b", {"a.py": "x = 1\n"}, ("a.py",))


def test_conforms_to_port():
    assert isinstance(FakeSearchTarget(), SearchTarget)


def test_deterministic_echo():
    ft = FakeSearchTarget()
    r1 = ft.run(FS, ["q1", "q2"], timeout_s=5)
    r2 = ft.run(FS, ["q1", "q2"], timeout_s=5)
    assert set(r1) == {"q1", "q2"}
    assert isinstance(r1["q1"], SearchResult)
    assert r1["q1"].context == r2["q1"].context        # deterministic
    assert "q1" in r1["q1"].context                     # echoes the query
    assert isinstance(r1["q1"].cost, CostMeter)


def test_canned_contexts():
    ft = FakeSearchTarget(contexts={"q1": "[doc:7] hello\n"})
    r = ft.run(FS, ["q1", "q2"], timeout_s=5)
    assert r["q1"].context == "[doc:7] hello\n"
    assert "q2" in r["q2"].context                       # falls back to echo


def test_varies_by_candidate():
    ft = FakeSearchTarget()
    other = FS.with_overlay({"a.py": "x = 2\n"})
    a = ft.run(FS, ["q"], timeout_s=5)["q"].context
    b = ft.run(other, ["q"], timeout_s=5)["q"].context
    assert a != b                                        # sha-tagged
