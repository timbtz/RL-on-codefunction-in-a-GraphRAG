# EvoRetrieve — pushing a self-improving optimizer from one function to a whole RAG system

> **A reflective, LLM-driven code-evolution optimizer applied to retrieval-augmented
> generation (RAG).** It starts by optimizing a single search function on a public
> benchmark, and is being extended toward a multi-component system where *ingestion* and
> *search* are co-optimized. The optimizer rewrites real source code (not a config),
> scores candidates with a judge-free exam, and prices every candidate in **real USD per
> query**.

<!-- Author note: working title "EvoRetrieve". This README is intentionally vision-first
     and light on internal mechanics — the internals are research-grade but not yet
     bulletproof, and they are not the point. The point is the research direction. -->

---

## TL;DR

A new class of system — **self-improving optimizers** that use an LLM as a *mutation
operator* over source code — has produced striking results on math and GPU kernels
([FunSearch](https://www.nature.com/articles/s41586-023-06924-6),
[AlphaEvolve](https://arxiv.org/abs/2506.13131),
[KernelEvolve](https://arxiv.org/abs/2512.23236)). These all share one shape: *propose a
code edit → evaluate it against an oracle → keep it if it wins → repeat.* They have been
pointed almost exclusively at problems with a cheap, exact oracle and a single objective
(speed, or a math score).

**EvoRetrieve asks how far that same optimizer goes when the target is a real agentic RAG
system** — where there is no exact oracle, where quality is semantic, and where every
query costs real money. Two design commitments make the problem tractable:

1. **A judge-free exam as the reward.** Quality is measured by deterministic
   multiple-choice exam accuracy (exact match, no LLM judge to game), with a
   *closed-book adjustment* that isolates what *retrieval* adds over the model's own
   parametric knowledge.
2. **Real USD/query as a first-class objective.** Every candidate is metered in dollars,
   and selection happens on the **accuracy–cost Pareto frontier**, not on accuracy alone.

The project is structured as a **trajectory**, not a single result:

| Phase | Target | What evolves | Status |
|---|---|---|---|
| **1 — single function** | one `search(query, graph)` function over a public KG benchmark (STARK) | one Python function | ✅ working, results below |
| **2 — whole system** | a real graph-retrieval service + its **ingestion** | multi-file source of a live service | 🚧 in progress — see [Roadmap](#roadmap) |

---

## Why this is interesting (research framing)

RAG quality is dominated by the retrieval stage, and the most consequential retrieval
decisions in a **graph** RAG system are *procedural* — how to seed entities, how far and
how selectively to traverse, when to spend an LLM call to expand or rerank, when to stop.
Those decisions live in **code**, not in a hyperparameter.

Today's automated RAG optimization mostly searches over **configurations** — it
recombines modules an engineer already wrote (e.g.
[AutoRAG](https://arxiv.org/abs/2410.20878); and, *per its public brief*, the ETH Agentic
Systems Lab's reasoning-agent AutoRAG, which proposes pipeline configs from exam
diagnoses). Code-evolution optimizers *can* synthesize genuinely new procedures — but
they've been applied to math and kernels, never to a RAG retrieval node against a live
graph database, and never with money as an objective.

**EvoRetrieve sits at that intersection** and treats it as an open question rather than a
solved one:

```mermaid
flowchart LR
    subgraph EU["edit unit →"]
      direction TB
      C["config / knobs"]
      P["prompt text"]
      K["source code"]
    end
    subgraph RW["reward →"]
      direction TB
      M["proxy metric"]
      J["LLM judge"]
      X["judge-free exam<br/>+ real $ cost"]
    end
    K -.->|"this project"| X
    style K fill:#0b6,stroke:#063,color:#fff
    style X fill:#0b6,stroke:#063,color:#fff
```

The lineage we build on (config search → prompt optimization → code evolution) is
surveyed in [related work](#related-work). The contribution we *claim* is deliberately
narrow and defensible: **applying reflective code evolution to a RAG retrieval system,
gated by a judge-free exam, with measured USD/query as a Pareto axis** — and then asking
how the optimizer behaves as the target grows from one function to a coupled
ingestion+search system.

---

## How the optimizer works

One optimization step is a closed reflective loop. The candidate is **real source code**;
the LLM is the mutation operator; reflection on *why queries failed* is the feedback
signal (a "textual gradient" in the [GEPA](https://arxiv.org/abs/2507.19457) sense).

```mermaid
flowchart TD
    A["Candidate program<br/>(real source code)"] --> B["Rollout<br/>run over a batch of exam queries<br/>in subprocess isolation"]
    B --> C["Reward<br/>judge-free MCQ accuracy<br/>+ closed-book adjustment<br/>+ metered USD / latency"]
    C --> D["Reflect<br/>attribute each miss:<br/>retrieval-reach vs ranking failure"]
    D --> E["Propose<br/>LLM emits multi-file<br/>SEARCH/REPLACE edits<br/>(edit budget + self-repair)"]
    E --> F{"Gate<br/>cost-aware:<br/>accuracy vs USD<br/>+ Pareto dominance"}
    F -->|accept| G["Pareto pool<br/>of incumbents"]
    F -->|reject| G
    G --> A
    G -.->|"after the run"| H["Held-out bake-off<br/>re-score survivors on a<br/>disjoint set → export best"]
```

The pieces, briefly (details in the per-package READMEs):

- **Candidate = code.** A candidate is an *overlay* of edited source files on an
  immutable base, materialized and run in a throwaway subprocess against a live graph
  database — so the optimizer edits the actual service, not a sandboxed toy.
- **Reward = judge-free exam.** Deterministic exact-match MCQ accuracy; a closed-book
  adjustment (`open-book correct − closed-book correct`) credits *retrieval*, not the
  model's memorized knowledge.
- **Feedback = reflection.** Each missed query is bucketed (did retrieval *reach* the
  answer at all, vs did it reach but *rank* it too low?) and fed back as text the next
  edit can act on.
- **Selection = cost-aware Pareto.** Acceptance blends accuracy against measured USD, and
  a Pareto pool keeps a *frontier* of incumbents rather than a single winner. A complexity
  / crash filter stops the optimizer from "winning" with degenerate programs.
- **Honesty = leakage control.** Per-run scores use small rotating gate sets; the number
  worth trusting is the **once-only held-out bake-off** on a disjoint 300-query set.

---

## Results

### Phase 1 — optimizing a single search function on STARK

Public benchmark: [STARK](https://stark.stanford.edu/) (semi-structured retrieval over a
knowledge graph). Metric: node-containment recall / hit / MRR via the STARK evaluator.
The optimizer rewrites one `search(query, graph)` function, starting from a simple
hand-written seed.

| Program | recall@20 | hit@1 | MRR | Evaluated on |
|---|---|---|---|---|
| Hand-written seed | 0.26 | 0.09 | 0.15 | gate set |
| **Optimized (best, exported)** | **0.44** | **0.28** | **0.35** | **300-query held-out** |
| | ×1.7 | ×3.1 | ×2.3 | |

The optimized program is the export from the 300-query held-out bake-off
(`runs/run7/select_holdout.json`); its held-out score matches its in-run score
(`export_changed = false`), i.e. **no overfitting gap** on this run. Absolute numbers on
STARK-prime are modest — the benchmark is hard and old — so the headline is the
**optimization gain from automated code evolution on a public set**, together with the
fact that every candidate was simultaneously **priced in dollars**.

> **Honesty notes (these matter for a research reader).**
> - Per-run gate sets are small (≤30 queries) and therefore noisy; only the 300-query
>   held-out number above should be read as a result.
> - This Phase-1 result was produced by the optimizer's original *single-function*
>   architecture. The Phase-2 *whole-service* extension is **not yet** producing a
>   non-regressing number — see Roadmap.

### Phase 2 — whole-system (ingestion + search) co-optimization

🚧 **In progress — to be updated.** The extension lets the optimizer edit a real
multi-file retrieval service and, ultimately, the **ingestion** that builds the graph it
searches — the interesting case where an ingestion change reshapes what search must do,
so the two have to be optimized *together*. The current multi-file target is being
stabilized (it regressed relative to Phase 1 — an instructive failure mode: code bloat
and crash-rate spirals under a weak complexity gate). **Results will be filled in here
once this is stable.**

<!-- UPDATE WHEN COMPLETE: replace this block with the Phase-2 results table
     (seed service vs co-optimized service: accuracy, USD/query, ingestion cost). -->

---

## Roadmap

- [ ] **Phase 2: stabilize the whole-service target** — fix the bloat/crash spiral (tighter
      complexity gate, stronger self-repair) so multi-file evolution matches or beats the
      single-function result. *(in progress)*
- [ ] **Co-optimize ingestion + search** — let the optimizer change how the graph is built,
      and measure the coupling between ingestion choices and downstream retrieval.
      ([`graphmod/`](graphmod) is the typed Neo4j ingestion framework this will plug into —
      today it is a clean schema/ingestion layer, *not yet* under optimizer control.)
- [ ] **Controlled science** — a like-for-like configuration-search baseline over the same
      knobs, a second corpus + held-out exam split, the cost-weight frontier sweep, and the
      reflection-on/off ablation. *(This is the empirical study, not yet run.)*

---

## Repository layout

| Path | What it is |
|---|---|
| [`graphretr-demo/`](graphretr-demo) | The optimizer core + Phase-1 STARK environment, runs, MLflow. See its README. |
| [`starksearch/`](starksearch) | Phase-2 whole-service STARK target (subprocess, multi-file). |
| [`graphsearch/`](graphsearch) | A second end-to-end agentic-retrieval target (graph QA). |
| [`graphmod/`](graphmod) | Typed, validated **Neo4j ingestion / schema framework** (TypeScript). Standalone today; the future ingestion-optimization target. |
| [`Orchestration/`](Orchestration) | Design docs / run plans. |

**Stack.** Python 3.11 (optimizer + targets), TypeScript/Node (`graphmod`). LLM backends
via OpenRouter (retrieval-time models + embeddings) and a configurable mutation backend.
Experiment tracking via MLflow.

> ⚠️ **Repo hygiene is mid-cleanup** — a tracked `.venv/`, caches, and a run graveyard are
> being removed (see the working cleanup plan). If you are reviewing this, the files worth
> reading are `graphretr-demo/src/graphretr_opt/optimizer/` (the loop, pool, and gate) and
> the per-package READMEs.

---

## Related work

The systems EvoRetrieve builds on and positions against:

- **Code evolution** — [FunSearch](https://www.nature.com/articles/s41586-023-06924-6)
  (Nature 2024), [AlphaEvolve](https://arxiv.org/abs/2506.13131) (DeepMind 2025),
  OpenEvolve, and [KernelEvolve](https://arxiv.org/abs/2512.23236) (Meta, kernel
  optimization). *Evolve source code; target math / kernels / speedup, not RAG, not $.*
- **Reflective prompt / pipeline optimization** —
  [GEPA](https://arxiv.org/abs/2507.19457) (reflective Pareto prompt evolution),
  [DSPy / MIPROv2](https://arxiv.org/abs/2406.11695), OPRO, TextGrad, Reflexion. *Optimize
  prompts / compound-AI parameters, not procedural retrieval code.*
- **Agent / skill design** — [Voyager](https://arxiv.org/abs/2305.16291) (lifelong skill
  library as code) and [ADAS](https://arxiv.org/abs/2408.08435) (automated design of
  agentic systems). *Closest in spirit to evolving a whole system.*
- **AutoML for RAG** — [AutoRAG](https://arxiv.org/abs/2410.20878) (greedy module/grid
  search) and the exam-generation evaluation of
  [Guinet et al.](https://arxiv.org/abs/2405.13622) (ICML 2024) that judge-free RAG scoring
  builds on. *Search over configurations of pre-built modules.*

A fuller, vendor-neutral survey of this landscape lives in the companion optimizer wiki.
