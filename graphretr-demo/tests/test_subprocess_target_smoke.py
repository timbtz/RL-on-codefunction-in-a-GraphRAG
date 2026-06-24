"""Integration smoke test for SubprocessSearchTarget -- real worktree overlay +
spawn + one query. Auto-skipped unless a live Neo4j and an LLM key are present
(GRAPHSEARCH_NEO4J_URL + OPENAI_API_KEY/ANTHROPIC_API_KEY), so the default test
run never needs infra.
"""
import os

import pytest

from graphretr_opt.artifact.file_set import FileSet
from graphretr_opt.env.search_target import SearchResult

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GRAPHSEARCH_NEO4J_URL")
         and (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))),
    reason="needs a live Neo4j (GRAPHSEARCH_NEO4J_URL) + an LLM API key")


def test_subprocess_one_query():
    from graphretr_opt.env.targets.subprocess_target import SubprocessSearchTarget
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = os.path.join(repo_root, "graphsearch", "src")
    opt_src = os.path.join(repo_root, "graphretr-demo", "src")
    editable = ("common/service/search/agentic_graph_traversal_search_service.py",)
    fs = FileSet.from_base(base, editable)
    target = SubprocessSearchTarget(
        neo4j_cfg={"url": os.environ["GRAPHSEARCH_NEO4J_URL"],
                   "username": os.environ.get("GRAPHSEARCH_NEO4J_USER", "neo4j"),
                   "password": os.environ.get("GRAPHSEARCH_NEO4J_PASSWORD", ""),
                   "database": os.environ.get("GRAPHSEARCH_NEO4J_DATABASE", "neo4j")},
        llm_cfg={"provider": os.environ.get("SEARCH_PROVIDER", "openai"),
                 "model": os.environ.get("SEARCH_MODEL", "gpt-4o-mini")},
        opt_src_dir=opt_src, service_relpath=editable[0])
    out = target.run(fs, ["Welcher Paketmanager wird verwendet?"], timeout_s=180)
    assert isinstance(out, dict) and len(out) == 1
    res = next(iter(out.values()))
    assert isinstance(res, SearchResult)
    # either we got context back or a contained error (never an uncaught crash)
    assert res.error is None or isinstance(res.error, str)
