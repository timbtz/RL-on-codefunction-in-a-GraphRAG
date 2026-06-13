"""Mutator -- the optimizer-model operator: render the rollout evidence, let the
agent (optionally) digest it, build the edit prompt, call the agent, apply the
edit, enforce the edit budget.

Two-phase, but agent-agnostic:
  1. evidence = the rendered failures / protected successes / dead-ends block.
  2. digest   = agent.digest(evidence)  -- SingleCoder returns it verbatim (one
     call total); TieredCoder runs a cheap Haiku pass that shrinks it.
  3. edits    = agent.complete(edit_prompt(digest), tier) as SEARCH/REPLACE
     blocks (full-module fenced block is the fallback).
Because both steps go through the same complete()/digest() seam, the
single-vs-tiered ablation never touches this file.

propose() returns (candidate SearchProgram or None, transcript). None means the
step is skipped (budget blown twice, unparseable reply, transient agent
failure). AgentUnavailable (hard CLI/spend limit) propagates so the loop can
stop the campaign instead of burning the remaining steps.
"""
from ..agents.single import AgentUnavailable
from ..artifact.program import SearchProgram
from .edits import EditError, apply_edits, extract_code, parse_edit_blocks


def _fmt_metrics(m):
    return (f"recall@20={m['recall@20']:.2f} hit@1={m['hit@1']:.0f} "
            f"hit@5={m.get('hit@5', 0.0):.0f} mrr={m['mrr']:.2f}")


def _format_failures(failures):
    if not failures:
        return "(none -- every rollout query had perfect recall@20 and hit@1)"
    parts = []
    for i, f in enumerate(failures, 1):
        sec = [f"[{i}] ({f['bucket']}) {f['query']}", f"    {_fmt_metrics(f['metrics'])}"]
        if f.get("error"):
            sec.append(f"    ERROR: {f['error']}")
        for nid, text in f["missed_gold"]:
            sec.append(f"    MISSED gold {nid} (absent from top-20): {text}")
        for rank, score, nid in f["gold_ranks"]:
            sec.append(f"    gold {nid} retrieved but ranked #{rank} (your score {score:.3f})")
        for nid, score, text in f["top_wrong"]:
            sec.append(f"    out-ranking NON-gold {nid} (your score {score:.3f}): {text}")
        parts.append("\n".join(sec))
    return "\n".join(parts)


def _format_wins(wins):
    if not wins:
        return ""
    lines = [f"- {w['query']}  ({_fmt_metrics(w['metrics'])})" for w in wins]
    return ("## Currently working -- do NOT regress these\n"
            + "\n".join(lines) + "\n\n")


def _format_buffer(entries):
    if not entries:
        return "(none yet)"
    return "\n".join(f"- step {e['step']}: {e['summary']}" for e in entries)


def format_evidence(failures, wins, buffer_entries):
    """The rollout evidence block -- what agent.digest() compresses."""
    return f"""## Worst failures from the latest rollout (train queries)
{_format_failures(failures)}

{_format_wins(wins)}## Previously rejected edits (did NOT improve the gate -- do not repeat)
{_format_buffer(buffer_entries)}"""


def build_prompt(incumbent_src, primitives_doc, digest, max_edits, safe_builtins):
    return f"""\
You are optimizing a Python retrieval program for STaRK-prime, a biomedical \
knowledge-graph QA benchmark. Given a natural-language query, the program must \
return candidate node ids ranked by relevance. The gate optimizes a blend of \
recall@20 (gold anywhere in top-20) AND ranking quality (hit@1 / MRR). Ranking \
is the current weakness: when gold IS retrieved but ranked below non-gold nodes, \
fix the SCORING, not the recall. Gold answer sets often contain MANY nodes \
(median ~10), of any node type.

## Current program (the incumbent)
```python
{incumbent_src}```

## Execution contract (hard-enforced by an AST gate -- violations are auto-rejected)
- The module must define `def search(q, G)`; helper functions and module-level
  constants are allowed. q is the query string, G the graph API below.
- Return a non-empty dict {{node_id(int): score(float)}}; higher score = more
  relevant. Only the ranking matters. Unreturned candidates rank last.
- NO imports. NO attribute names starting with underscore. Only these
  builtins: {sorted(safe_builtins)}.
- Each graph call has a 2 s server-side timeout; primitives cap fan-out.
  A query crashing or timing out zeroes that query's score.

## Graph API
{primitives_doc}

## Evidence (latest rollout)
{digest}

## Your task
1. Diagnose in a few sentences WHY the incumbent misses or mis-ranks these gold
   nodes (look at what the missed/out-ranking node texts share and which
   primitives could reach or re-rank them).
2. Propose your change as up to {max_edits} SEARCH/REPLACE blocks. Each block:
   <<<<<<< SEARCH
   <exact contiguous lines copied verbatim from the incumbent>
   =======
   <the replacement lines>
   >>>>>>> REPLACE
   The SEARCH text must match the incumbent EXACTLY and appear exactly once.
   Prefer small, targeted edits. Give a one-line rationale per block.
   (Only if a wholesale rewrite is unavoidable, you may instead end with the
   COMPLETE module in ONE ```python block -- but edit blocks are preferred.)
"""


class Mutator:
    def __init__(self, agent, primitives_doc, safe_builtins):
        self._agent = agent
        self._doc = primitives_doc
        self._safe_builtins = safe_builtins

    @property
    def call_counts(self):
        """Per-model call tally (for MLflow cost reporting); {} if unsupported."""
        return getattr(self._agent, "call_counts", {})

    def _candidate(self, program, resp):
        """Apply the response. -> (SearchProgram, n_edits) or (None, 0) if the
        reply yields no usable program. EditError on blocks falls back to the
        full-module path before giving up."""
        blocks = parse_edit_blocks(resp)
        if blocks:
            try:
                return program.with_src(apply_edits(program.src, blocks)), len(blocks)
            except EditError:
                pass  # malformed/anchorless blocks -> try a full module instead
        try:
            cand = program.with_src(extract_code(resp))
        except ValueError:
            return None, 0
        return cand, program.edit_distance(cand)

    def propose(self, program: SearchProgram, failures, wins, buffer_entries,
                edit_budget, plateau=False):
        """-> (candidate SearchProgram or None, transcript:str)."""
        evidence = format_evidence(failures, wins, buffer_entries)
        transcript = [f"# EVIDENCE\n\n{evidence}"]
        try:
            digest = self._agent.digest(evidence)
        except AgentUnavailable:
            raise
        except Exception as e:           # analyst hiccup -> use raw evidence
            transcript.append(f"# ANALYST FAILED -- using raw evidence\n\n{e}")
            digest = evidence
        if digest is not evidence:
            transcript.append(f"# DIGEST (analyst)\n\n{digest}")

        prompt = build_prompt(program.src, self._doc, digest, edit_budget,
                              self._safe_builtins)
        transcript.append(f"# PROMPT\n\n{prompt}")
        tier = "architect" if plateau else "editor"
        for attempt in (1, 2):
            try:
                resp = self._agent.complete(prompt, tier=tier)
            except AgentUnavailable:
                raise
            except Exception as e:
                transcript.append(f"# AGENT CALL FAILED (attempt {attempt})\n\n{e}")
                return None, "\n\n---\n\n".join(transcript)
            transcript.append(f"# RESPONSE ({tier}, attempt {attempt})\n\n{resp}")

            cand, n_edits = self._candidate(program, resp)
            if cand is None:
                prompt += ("\n\nYour previous reply contained neither a usable "
                           "SEARCH/REPLACE block (the SEARCH text must match the "
                           "incumbent exactly) nor a ```python module. Reply again.")
                continue
            if n_edits <= edit_budget:
                return cand, "\n\n---\n\n".join(transcript)
            prompt += (f"\n\nYour previous candidate changed {n_edits} regions but "
                       f"the budget is {edit_budget}. Keep your best {edit_budget} "
                       "edits and reply again.")
        transcript.append("# SKIPPED: no in-budget candidate after 2 attempts")
        return None, "\n\n---\n\n".join(transcript)
