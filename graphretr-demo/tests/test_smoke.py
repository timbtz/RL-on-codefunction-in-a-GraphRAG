#!/usr/bin/env python
"""test_smoke.py -- structural smoke test for the curated layout.

Exercises every primitive against the live graph and every sandbox wall, plus
the new module seams (backend swap point, MetricVector, Pareto dominance,
EditBudget schedules, SearchProgram diff/edit-distance). Run directly:

    .venv/bin/python -m tests.test_smoke      # from project root
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from graphretr_opt.config import load_config
from graphretr_opt.env.backends.falkordb import FalkorDBBackend
from graphretr_opt.env.cache import PrimitiveCache
from graphretr_opt.env.embedder import make_embedder
from graphretr_opt.env.openai_client import OpenAIBudget
from graphretr_opt.env.retrieval_graph import RetrievalGraph
from graphretr_opt.env.sandbox import Sandbox, SandboxError
from graphretr_opt.artifact.program import SearchProgram
from graphretr_opt.reward.objectives import MetricVector, code_complexity
from graphretr_opt.reward.pareto import dominates
from graphretr_opt.optimizer.edit_budget import EditBudget

cfg = load_config()
backend = FalkorDBBackend(cfg.falkor_host, cfg.falkor_port, cfg.graph_name)
# Mirror campaign.py: use the SAME embedder the graph is indexed with (config
# selects openai_small=1536-d vs minilm=384-d). A hardcoded QueryEmbedder() here
# was 384-d and tripped "Vector dimension mismatch" against the 1536-d index.
# openai_small needs a metered budget; build one when any key (OpenAI OR
# OpenRouter) is present, else the minilm path stays key-free as before.
budget = None
if (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")):
    budget = OpenAIBudget(os.path.join(cfg.runs_dir, "openai_usage.json"),
                          ceiling_usd=cfg.openai_budget_usd)
G = RetrievalGraph(cfg, backend, PrimitiveCache(),
                   make_embedder(cfg, budget), llm_budget=budget)
sandbox = Sandbox(G, default_timeout_s=cfg.probe_timeout_s)
checks = 0


def ok(name, detail=""):
    global checks
    checks += 1
    print(f"[PASS] {name}{': ' + str(detail) if detail else ''}")


# ===================== env: primitives =====================
assert len(G.labels) == 10 and len(G.rel_types) == 18 and "gene/protein" in G.ntypes
ok("allowlists", f"{len(G.labels)} labels / {len(G.rel_types)} rels / {len(G.ntypes)} ntypes")

for q in ("What disease is associated with the BRCA1 gene?",
          "Which drugs target the EGFR receptor tyrosine kinase?"):
    hits = G.vector_search(q, k=30)
    sims = [s for _, s in hits]
    assert len(hits) == 30 and all(0.0 <= s <= 1.0 for s in sims)
    assert sims == sorted(sims, reverse=True) and all(isinstance(i, int) for i, _ in hits)
ok("vector_search all-label", f"top sim {sims[0]:.3f}")
hits1 = G.vector_search("breast cancer", k=5, label="disease")
ids = [i for i, _ in hits1]
ok("vector_search single label", hits1[:2])

texts = G.get_text(ids)
assert set(texts) == set(ids) and all(isinstance(t, str) and t for t in texts.values())
ok("get_text", list(texts.values())[0][:50])

nbrs = G.get_neighbors(ids, limit=40)
assert nbrs and len(nbrs) <= 40 and all(isinstance(s, int) for s, r, d in nbrs)
ok("get_neighbors out", f"{len(nbrs)} edges")
ok("get_neighbors in (single id)", f"{len(G.get_neighbors(ids[0], direction='in', limit=10))} edges")

assert set(G.filter_nodes(ids, ntype="disease")) == set(ids)
assert G.filter_nodes(ids, text_contains="zzzqqq_nonexistent") == []
ok("filter_nodes", "disease kept, junk substring -> 0")

exp = G.k_hop_expand(ids[:2], k=2, max_nodes=150)
assert 0 < len(exp) <= 150 and len(exp) == len(set(exp))
ok("k_hop_expand", f"{len(exp)} nodes / 2 hops")
ok("k_hop_expand rel-typed",
   f"{len(G.k_hop_expand(ids[:2], k=1, rel_type='associated_with', max_nodes=150))} via associated_with")

a, b = nbrs[0][0], nbrs[0][2]
path = G.shortest_path(a, b, max_len=99)  # clamps to 3 (SPpaths runaway wall)
assert path and path[0] == a and path[-1] == b
ok("shortest_path (max_len capped at 3)", path)

ranked = G.rank_by_text(exp, "breast cancer susceptibility", top=10)
rsims = [s for _, s in ranked]
assert 0 < len(ranked) <= 10 and rsims == sorted(rsims, reverse=True)
ok("rank_by_text", f"top {ranked[0]}")

assert len(G.vector_search("cancer", k=99999)) <= cfg.max_fanout
for bad in (lambda: G.get_neighbors(ids, rel_type="DROP TABLE"),
            lambda: G.vector_search("x", label="bogus"),
            lambda: G.filter_nodes(ids, ntype="bogus"),
            lambda: G.k_hop_expand(ids, k="2"),
            lambda: G.get_text([1.5])):
    try:
        bad(); raise AssertionError("junk arg accepted")
    except ValueError:
        pass
ok("caps + junk args rejected")

t0 = time.time(); r1 = G.vector_search("cancer", k=50); cold = time.time() - t0
t0 = time.time(); r2 = G.vector_search("cancer", k=50); warm = time.time() - t0
assert r1 == r2 and warm < cold
r2.append(("tamper", 0))
assert G.vector_search("cancer", k=50) == r1
ok("memo cache", f"cold {cold*1000:.0f}ms -> warm {warm*1000:.1f}ms, tamper-proof")
assert G.get_text(ids) is not None and backend.query_count > 0
ok("backend.query_count wired", backend.query_count)

# ===================== env: sandbox =====================
seed = SearchProgram.from_file(
    os.path.join(ROOT, "src/graphretr_opt/artifact/seeds/vector_only.py"), "vector_only")
fn = sandbox.compile(seed.src)
pred, stats = sandbox.run(fn, "What disease is associated with the BRCA1 gene?")
assert len(pred) == 100 and stats.queries > 0 and stats.latency_s > 0
ok("sandbox run -> (pred, RunStats)", f"{len(pred)} ids, {stats.queries} queries")

for name, bad_src in (
    ("import", "import os\ndef search(q, G):\n    return {1: 1.0}\n"),
    ("dunder attr", "def search(q, G):\n    return {len(q.__class__.__name__): 1.0}\n"),
    ("private attr", "def search(q, G):\n    G._backend.ro_query('MATCH (n) DELETE n')\n    return {1: 1.0}\n"),
    ("unknown name", "def search(q, G):\n    return {1: float(getattr(G, 'x'))}\n"),
    ("wrong signature", "def search(query):\n    return {1: 1.0}\n"),
    ("exec call", "def search(q, G):\n    exec('x=1')\n    return {1: 1.0}\n"),
):
    try:
        sandbox.compile(bad_src); raise AssertionError(f"accepted {name}")
    except SandboxError:
        pass
ok("sandbox: 6 malicious/malformed rejected")

slow = "def search(q, G):\n    x = 0\n    while True:\n        x = x + 1\n    return {1: float(x)}\n"
try:
    sandbox.probe(sandbox.compile(slow), ["test"], timeout_s=1.5)
    raise AssertionError("accepted infinite loop")
except SandboxError as e:
    ok("sandbox: probe timeout kills infinite loop", str(e)[:50])

# ===================== reward + optimizer units =====================
hi = MetricVector(quality={"recall@20": 0.30, "hit@1": 0.1, "mrr": 0.2}, latency_s=1.0, db_load=5)
lo = MetricVector(quality={"recall@20": 0.20, "hit@1": 0.0, "mrr": 0.1}, latency_s=2.0, db_load=9)
assert dominates(hi, lo) and not dominates(lo, hi)
ok("Pareto dominance", "hi > lo on every axis")

eb_c = EditBudget("const", 4, 1, 30); eb_cos = EditBudget("cosine", 4, 1, 30)
assert eb_c.L_t(0) == 4 == eb_c.L_t(29)
assert eb_cos.L_t(0) == 4 and eb_cos.L_t(29) <= 2
ok("EditBudget schedules", f"cosine 0->{eb_cos.L_t(0)}, 29->{eb_cos.L_t(29)}")

p2 = seed.with_src(seed.src.replace("k=100", "k=50"))
assert seed.edit_distance(p2) >= 1 and "k=100" in seed.diff(p2)
assert code_complexity(seed.src) > 0
ok("SearchProgram edit-distance + diff", f"L={seed.edit_distance(p2)}")

print(f"\n==== {checks}/{checks} structural checks PASSED ====")
