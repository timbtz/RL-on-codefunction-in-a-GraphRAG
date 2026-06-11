---
title: "Demo build guide — Stage 1 on a Hetzner VPS"
source: []
created: 2026-06-10
updated: 2026-06-10
tags: [changelog, plan, demo, runbook, project]
connects: [optimizer-project-plan, stark-substrate, graph-db-setup, graphretr-pipeline, coding-agent-search-optimizer]
---

# Demo build guide — Stage 1 on a Hetzner VPS

Concrete runbook for the **Stage 1 demo** of [[optimizer-project-plan]]: a coding agent optimizes a graph-retrieval program over STaRK-prime in FalkorDB, scored by node-containment, logged to MLflow — the artifact shown to the ETH lab. Component detail: [[eval-and-infra/stark-substrate]] · [[eval-and-infra/graph-db-setup]] · [[eval-and-infra/graphretr-pipeline]]. Target: a few person-days, prime only.

## Cost & footprint
**Free:** STaRK (MIT; HF download), FalkorDB (self-host Docker; accessed via local Redis protocol — not a paid API), MLflow (self-host), the VPS (already owned). **Paid:** only LLM tokens for the coding-agent + reflection calls (single-digit–low-tens of $ for a prime run). **Embeddings:** one-time; free with local `sentence-transformers` (CPU) or ~$0.15 via OpenAI `text-embedding-3-small` for all ~129K prime nodes. **RAM:** prime (129K nodes/8M edges) + small embeddings fits in ~4–8 GB → a Hetzner **CPX31/CPX41** suffices. No GPU (coding agent = API call; embeddings = CPU).

## Stack
Docker: **FalkorDB** (graph+vector) + **MLflow** tracking server. Python venv: `stark-qa`, `falkordb`, `sentence-transformers` (or `openai`), `mlflow`, and your agent SDK (`anthropic`/`openai`).

## Step 0 — provision the VPS
```bash
# Hetzner Ubuntu box
curl -fsSL https://get.docker.com | sh
apt install -y python3-venv build-essential
python3 -m venv ~/opt && source ~/opt/bin/activate
pip install stark-qa falkordb sentence-transformers mlflow anthropic openai
docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest
docker run -d --name mlflow -p 5000:5000 -v ~/mlruns:/mlruns ghcr.io/mlflow/mlflow:latest \
  mlflow server --host 0.0.0.0 --backend-store-uri sqlite:////mlruns/mlflow.db \
  --default-artifact-root /mlruns
```
(Bind 6379/5000 to localhost or firewall them; FalkorDB browser UI on :3000.)

## Step 1 — STaRK env
```python
from stark_qa import load_qa, load_skb
skb = load_skb('prime', download_processed=True)   # ~5 min, free, from HuggingFace
qa  = load_qa('prime')                              # queries + gold answer_ids
split = qa.get_idx_split()                          # {'train','val','test'} index lists
```
Inspect: `skb.node_types`, `skb.get_tuples()` (schema triples), `skb.get_doc_info(i)` (node text), `skb.get_neighbor_nodes(i, rel)`.

## Step 2 — ETL into FalkorDB (one-time)
Map node-type→label, relation→typed edge, node text→`text` property, embedding→`embedding` property + vector index ([[graph-db-setup]]). Embed once:
```python
from sentence_transformers import SentenceTransformer
from falkordb import FalkorDB
enc = SentenceTransformer('all-MiniLM-L6-v2')              # 384-d, CPU, free
g = FalkorDB(host='localhost', port=6379).select_graph('prime')
# nodes: batch CREATE with label=node_type, props {id, text, embedding}
# edges: MATCH/CREATE typed rels from skb.edge_index + edge_type_dict
g.query("CREATE VECTOR INDEX FOR (n:gene) ON (n.embedding) OPTIONS {dimension:384, similarityFunction:'cosine'}")
```
(For ~129K nodes batch in chunks; persist so you only ETL once.)

## Step 3 — the primitive service (`RetrievalGraph`)
A read-only class wrapping the FalkorDB driver; each primitive = one **parameterized, capped, read-only Cypher** query (production-shaped, *not* STaRK's in-memory API — see [[graph-db-setup]]). Memoize by (method, args).
```python
class RetrievalGraph:
    def vector_search(self, text, k=20, label=None): ...   # ANN over vector index -> [id]
    def get_neighbors(self, ids, rel_type=None, direction='out', limit=50): ...
    def k_hop_expand(self, ids, k, rel_filter=None, max_nodes=200): ...
    def filter_nodes(self, ids, predicate): ...
    def shortest_path(self, a, b, max_len=4): ...
    def rank_by(self, ids, key, top=10): ...
```
All issue `graph.ro_query(...)` with `TIMEOUT`; every primitive returns node-ids.

## Step 4 — seed program + scorer
The **parent search program** the optimizer mutates (layer 2 in [[coding-agent-search-optimizer]]) — start deliberately naive so there's visible headroom:
```python
def search(q, G):                       # SEED: vector-only baseline
    return {nid: s for nid, s in G.vector_search(q, k=20)}   # node_id -> score
```
Scorer wires to STaRK's evaluator:
```python
from stark_qa.evaluator import Evaluator
ev = Evaluator(skb.candidate_ids)
def score(program, idxs):
    rec = []
    for i in idxs:
        q, gold = qa[i].query, qa[i].answer_ids
        pred = program(q, G)            # node_id -> score
        rec.append(ev.evaluate(pred, gold, metrics=['hit@1','recall@20','mrr']))
    return mean(rec)                    # the reward
```

## Step 5 — the optimizer loop (minimal SkillOpt-style)
For the demo, **hand-roll a thin loop** — it directly demonstrates *your* code-opt approach to the lab and is ~150 lines; swap in [[research-foundations/gepa|GEPA/DSPy]] later for Pareto + autolog. Mechanics from [[research-foundations/skillopt-method|SkillOpt]]:
```python
prog = SEED; best = score(prog, split['val']); buffer = []
for step in range(N):
    # 1. rollout: run prog on a train batch, collect failing queries + retrieved-vs-gold
    fails = rollout(prog, sample(split['train']))
    # 2. reflect: coding-agent LLM proposes <=L bounded edits to prog source,
    #    given fails + the rejected-edit buffer (negative replay)
    cand = llm_edit(prog, fails, buffer, max_edits=4)      # the "RL" step
    # 3. validate gate
    s = score(cand, split['val'])
    if s > best:  prog, best = cand, s                     # accept (strict-greater)
    else:         buffer.append((cand_diff, best - s))     # reject -> negative replay
    mlflow.log_metric('val_recall@20', s, step=step)
# TEST scored ONCE at the end, separate run
```
LLM calls = the only cost; the coding agent (e.g. claude-opus) is the program mutator, a second call does the reflection/diagnosis.

## Step 6 — MLflow wiring
`mlflow.set_tracking_uri('http://localhost:5000')`; experiment `graphretr-opt/demo`; parent run per campaign, log params (max_edits, models, k), stepped `val_recall@20`/`hit@1`/`mrr` curves, artifacts (`best_search.py`, the rejected buffer, reflection transcripts). Optional `@mlflow.trace` on `llm_edit` + the primitives to see the evolving code + retrieved node-sets ([[graphretr-pipeline]]). Test metrics logged once in a separate `final-test-report` run.

## Step 7 — what to show the ETH lab
1. **The before/after curve** — seed (vector-only) Recall@20 vs the agent-evolved hybrid program, on the locked test split.
2. **The evolved code** — diff the seed vs `best_search.py`: the agent *discovered* a traverse-then-filter strategy (vector seed → typed-edge expand → text re-rank). Concrete, legible "the AI made the search smarter."
3. **The environment** — MLflow curves + traces (rollouts, edits, gate accept/reject, the rejected buffer working).
4. **The framing facts** — program-space specialization, **no model finetuning**, a frozen model + an evolving search program; transferable to any KG (the Reply-KG / thesis story). Design: [[coding-agent-search-optimizer]]; positioning: [[graphrag-search-optimization]].

## Gotchas
- **Stage 0 first** (optional but ideal): one-shot-LLM-rewrite vs the seed — confirms headroom exists before investing ([[optimizer-project-plan]]).
- Persist the ETL (don't re-embed each run); cache primitive calls (huge inner-loop speedup).
- Keep the seed *naive* so improvement is visible; firewall the VPS ports.

## See also
- [[optimizer-project-plan]] (where this is Stage 1) · [[coding-agent-search-optimizer]] (the 3 layers / two-timescale design) · [[eval-and-infra/graphretr-pipeline]] (v0→v1 to Databricks).
