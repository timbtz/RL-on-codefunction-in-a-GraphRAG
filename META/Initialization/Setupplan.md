---
title: "Optimizer agent brief — context + prompt for the Stage-1 loop"
source: []
created: 2026-06-10
updated: 2026-06-10
tags: [plan, demo, prompt, optimizer, runbook, project]
connects: [demo-build-guide, optimizer-project-plan, coding-agent-search-optimizer, skillopt-method, gepa, stark-substrate, graph-db-setup, graphretr-pipeline]
---

# Optimizer agent brief — context + prompt for the Stage-1 loop

Hand-off for the coding agent building the **optimizer** half of the Stage-1 demo (the env — FalkorDB + STaRK ETL + MLflow — is being stood up per [[demo-build-guide]]). This page is the *why* + the *contract* + a paste-ready reflection prompt. Design authority: [[coding-agent-search-optimizer]]; loop mechanics: [[research-foundations/skillopt-method|SkillOpt]]; future harness: [[research-foundations/gepa|GEPA]].

## Mission (one sentence)
Keep the LLM **frozen**; evolve an executable **graph-retrieval program** so its node-containment score on held-out STaRK-prime queries beats a deliberately naive seed — demonstrating *program-space* specialization with **no model finetuning**. Why this and not RL: enterprise KGs have no shared schema, so a finetuned model is re-paid per KG; editing search *code* needs no GPUs, no labelled trajectories, no retrain — the cheaper specialization vehicle. The artifact shown to the ETH lab is the **optimizer + its before/after curve**, not the search function.

## What we take from SkillOpt (and what we don't)
SkillOpt's transferable contribution is **not "edit prose"** — it is a complete optimizer for a non-differentiable artifact, and the machinery is **artifact-agnostic**. The artifact swaps from `best_skill.md` → `best_search.py`; the optimizer model becomes a **coding agent** that reads failed retrieval trajectories and emits code diffs. Four mechanisms carry over — implement all four in the demo loop:

| SkillOpt mechanism | Here, in the demo |
|---|---|
| **Bounded per-step edit budget `L_t`** (the learning-rate analogue) | The reflection step emits **≤4 add/delete/replace edits** to the program source — not a free rewrite. Big edits early is fine; this is the single most load-bearing control. |
| **Train / select / test split + validation gate** (propose-and-test, not self-edit) | Accept a candidate **only if** its val score strictly exceeds the incumbent. Locks out gameable rewrites. |
| **Rejected-edit buffer** (negative replay, zero inference cost) | A gate-failing candidate's diff + its score drop is appended to a buffer fed into the next reflection prompt: *"these edits were tried and made it worse — don't repeat."* |
| **Slow/momentum field** | *Skip for the demo* (Stage 2). A protected longitudinal-guidance block; not needed for ~150 lines. |

## Where GEPA fits — NOT yet
For the demo, **hand-roll the thin loop** (~150 lines). It directly demonstrates *our* approach to the lab and avoids a framework dependency. **Do not install `dspy`/`gepa` for Stage 1.** [[research-foundations/gepa|GEPA]] is the Stage-2 swap-in: it already provides Pareto multi-objective selection, scalar+textual-trace feedback, and arbitrary code-as-candidate, so Stage 2 = *fork GEPA's adapter harness + graft SkillOpt's bounded-edits / rejected-buffer / momentum on top*. Build the demo loop with that seam in mind (keep `score()` returning a metric **dict**, not a pre-collapsed scalar — see Reward), but ship the hand-rolled version now.

## What is mutable vs fixed (three layers)
Mutate **only layer 2.** Full rationale: [[coding-agent-search-optimizer]] §three layers.
1. **Primitives** — `vector_search`, `get_neighbors`, `k_hop_expand`, `filter_nodes`, `shortest_path`, `rank_by`, set-ops. The fixed, vetted, read-only instruction set, wrapping **parameterized Cypher over the live FalkorDB** ([[graph-db-setup]]). The agent never edits these and never emits raw Cypher.
2. **The parent search program** — the Python function that *composes* the primitives: hop order, vector-seed vs edge-expand, fan-out caps, prune thresholds, text re-rank, evidence fusion. **This is the only thing the loop rewrites.**
3. **Primitive set / strategy family** — slow-loop, Stage 2 only.

## Q/A + validation-set wiring (the data contract)
STaRK ships everything; do not synthesize questions for the demo.
```python
from stark_qa import load_qa, load_skb
qa    = load_qa('prime')              # queries + gold answer_ids (node-ID SETS)
split = qa.get_idx_split()            # {'train': [...], 'val': [...], 'test': [...]}
```
Discipline — map the splits to SkillOpt roles and **never cross them**:
- **`train` = rollout evidence.** Sample a batch each step; run the program; collect the **failing** queries (low containment) + their retrieved-vs-gold node-IDs. This is what the reflection reads. *Never gates.*
- **`val` (= SkillOpt `D_sel`) = the gate.** Score every candidate here; strict-greater acceptance. To stay cheap, gate on a **fixed subsample** (e.g. 200 val queries) held constant across the whole campaign (so step-to-step deltas are comparable).
- **`test` = LOCKED.** Scored **exactly once**, after the loop, in a separate MLflow run. Never read inside the loop. (Ideally behind a path the loop code can't touch.)

Scorer = STaRK's deterministic `Evaluator` — the reward is **node-containment**, attributable to the search alone (no LLM judge):
```python
from stark_qa.evaluator import Evaluator
ev = Evaluator(skb.candidate_ids)
def score(program, idxs):              # returns a METRIC DICT (keep it a dict for the GEPA seam)
    rows = []
    for i in idxs:
        pred = program(qa[i].query, G)          # {node_id: float}
        rows.append(ev.evaluate(pred, qa[i].answer_ids,
                                metrics=['hit@1','recall@20','mrr']))
    return {k: mean(r[k] for r in rows) for k in rows[0]}   # gate on recall@20
```
Gate metric for the demo: **`recall@20`** (prime's best-reported is 0.381 — large headroom). Log `hit@1`/`mrr` alongside.

## The seed search the optimizer iterates over
Start **deliberately naive** so improvement is visible (vector-only ignores the typed-edge topology STaRK gold requires — guaranteed headroom):
```python
def search(q, G):                      # SEED — layer-2 artifact the loop mutates
    hits = G.vector_search(q, k=20)    # ANN over node embeddings only
    return {nid: s for nid, s in hits} # node_id -> score
```
The win the agent should *discover* (don't hand it this): a **hybrid traverse-then-filter** program — vector seed → typed-edge expand (`get_neighbors`/`k_hop_expand`) → text re-rank — because STaRK gold demands a relational **and** a textual constraint ([[stark-substrate]]). That diff (seed → evolved) is demo exhibit #2.

## The loop (assemble these pieces)
```
prog = SEED; best = score(prog, VAL_SUBSAMPLE); buffer = []
for step in range(N):
    fails = rollout(prog, sample(split['train'], B))     # retrieved-vs-gold for missed queries
    cand  = llm_edit(prog, fails, buffer, max_edits=4)    # reflection -> ≤4 bounded edits
    s     = score(cand, VAL_SUBSAMPLE)
    if s['recall@20'] > best['recall@20']:
        prog, best = cand, s                              # ACCEPT (strict-greater)
    else:
        buffer.append({'diff': diff(prog, cand),
                       'drop': best['recall@20'] - s['recall@20']})   # REJECT -> negative replay
    mlflow.log_metric('val_recall@20', s['recall@20'], step=step)
# after loop: score(prog, split['test']) ONCE, in a separate 'final-test-report' run
```
MLflow per [[graphretr-pipeline]]: parent run per campaign; params (`max_edits`, model names, `k`, split sizes); stepped `val_recall@20`/`hit@1`/`mrr`; artifacts (`best_search.py`, the rejected buffer, reflection transcripts). LLM calls are the only cost — cache primitive calls by `(method,args)` so the loop stays LLM-bound, not DB-bound.

## The reflection prompt (paste-ready — this is the "optimizer prompt")
Two LLM roles per step: a **diagnosis/edit** call (below) is the mutator; you may split diagnosis and editing into two calls, but one combined call is fine for the demo.

> **System:** You are a retrieval-program optimizer. You improve a Python function `search(q, G)` that retrieves node IDs from a biomedical knowledge graph (STaRK-prime) to maximize **recall@20** against a gold set of node IDs. You may ONLY recompose the fixed, read-only primitives exposed by `G` (`vector_search`, `get_neighbors`, `k_hop_expand`, `filter_nodes`, `shortest_path`, `rank_by`, set-ops — signatures below). You may NOT write raw Cypher, import libraries, call the network, or modify the primitives. Every primitive returns node IDs and takes an explicit cap argument; keep caps bounded.
>
> **Context the model receives each step:**
> 1. The current `search` source.
> 2. The primitive signatures + one-line semantics (from [[graph-db-setup]]).
> 3. **Failing queries** from this step's train rollout: the query text, the gold node IDs, the IDs your program retrieved, and which gold IDs were missed. (Gold answers require BOTH a relational hop and a textual match — diagnose which half failed.)
> 4. The **rejected-edit buffer**: diffs tried before that lowered the score — do not repeat them.
>
> **Task:** Diagnose *why* the misses happened (wrong/absent hop, over-broad fan-out, missing text filter, bad ranking), then return **at most 4** localized edits to the function (add/delete/replace), as a unified diff plus the full new function body. Make the smallest change that addresses the diagnosed failure mode. Do not rewrite wholesale. Explain each edit in one line.

## Definition of done (what proves the demo)
1. **Before/after curve** — seed vs evolved `recall@20` on the **locked test split**, one number each, plus the val curve over steps.
2. **The evolved code** — `diff(SEED, best_search.py)` showing the agent discovered hybrid traverse-then-filter.
3. **The environment working** — MLflow showing rollouts, edits, gate accept/reject, and the rejected buffer actually suppressing repeats.
4. **Framing** — frozen model + evolving search program; transferable to any KG.

## Pitfalls (read before coding)
- **Run Stage 0 first** (optional but ideal): one frontier-LLM one-shot rewrite of the seed vs the seed. If a single shot already closes the gap, the prize is in schema/policy, not the optimizer — re-scope before building the loop ([[optimizer-project-plan]]).
- Keep the **gate subsample fixed** across the campaign, or step deltas are noise.
- Keep the seed **naive** — a strong seed hides the improvement.
- `score()` returns a **dict**, never a pre-collapsed scalar (Stage-2 GEPA Pareto needs the vector).
- **Never** let test leak into the loop.

## External references (for the agent — these resolve outside the vault)
Stage-1 essentials:
- **STaRK** — paper arXiv [2404.13207](https://arxiv.org/abs/2404.13207); repo [github.com/snap-stanford/stark](https://github.com/snap-stanford/stark); PyPI `stark-qa`; processed SKBs on HF [`snap-stanford/stark`](https://huggingface.co/datasets/snap-stanford/stark). API you need: `load_qa`, `load_skb`, `qa.get_idx_split()`, `Evaluator(candidate_ids).evaluate(pred_dict, answer_ids, metrics=[...])`.
- **FalkorDB** — repo [github.com/FalkorDB/FalkorDB](https://github.com/FalkorDB/FalkorDB); bulk loader [github.com/FalkorDB/falkordb-bulk-loader](https://github.com/FalkorDB/falkordb-bulk-loader). Use `GRAPH.RO_QUERY` + per-query `TIMEOUT`; build the primitive layer as **parameterized Cypher**, bypassing FalkorDB's NL→Cypher GraphRAG-SDK.
- **STaRK-Prime → graph-DB loader (reference ETL)** — [github.com/neo4j-product-examples/neo4j-gnn-llm-example](https://github.com/neo4j-product-examples/neo4j-gnn-llm-example) (loads STaRK-Prime into Neo4j; adapt the node/edge mapping for FalkorDB).
- **MLflow GenAI tracing** — [mlflow.org/docs/latest/genai/tracing](https://mlflow.org/docs/latest/genai/tracing/) (`@mlflow.trace`, `mlflow.<lib>.autolog()`, nested runs).
- **SkillOpt** — the loop we're porting: paper arXiv 2605.23904 (v2); repo `microsoft/SkillOpt` (MIT — read it for the bounded-edit / rejected-buffer / gate implementation, *don't* depend on it).

Stage-2 only (do **not** install for the demo — context for the seam):
- **GEPA** — paper arXiv [2507.19457](https://arxiv.org/abs/2507.19457) (ICLR 2026 oral); repos [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa) and `dspy.GEPA` ([dspy.ai/api/optimizers/GEPA](https://dspy.ai/api/optimizers/GEPA/overview/)); DSPy+MLflow autolog tutorial [dspy.ai/tutorials/optimizer_tracking](https://dspy.ai/tutorials/optimizer_tracking/). The `GEPAAdapter` (`evaluate()` + `make_reflective_dataset()`) is what we'd implement in Stage 2.
- **OpenEvolve** (open AlphaEvolve reimpl) — borrow its cascade-evaluation pattern (cheaply reject programs that error / retrieve nothing before full scoring). **ADAS** (arXiv [2408.08435](https://arxiv.org/abs/2408.08435)), **Hyperband** (arXiv [1603.06560](https://arxiv.org/abs/1603.06560)), **MAP-Elites** (arXiv [1504.04909](https://arxiv.org/abs/1504.04909)) — the two-timescale strategy-switching layer, not the demo.

## See also
- [[demo-build-guide]] — the env this loop runs on (the Steps 0–7 runbook). · [[optimizer-project-plan]] — where this sits (Stage 1).
- [[coding-agent-search-optimizer]] — full design (three layers, two timescales, multi-objective reward). · [[research-foundations/skillopt-method]] — the four mechanisms. · [[research-foundations/gepa]] — the Stage-2 harness.
- [[stark-substrate]] · [[graph-db-setup]] · [[graphretr-pipeline]] — substrate, primitives, observability.
