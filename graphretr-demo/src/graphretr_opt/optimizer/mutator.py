"""Mutator -- SkillOpt's optimizer-model operator: build the reflection prompt,
call the optimizer agent, extract the candidate, enforce the edit budget.

propose() returns (candidate_src or None, transcript). None means the step is
skipped (budget blown twice, unparseable reply, or agent failure) -- the loop
treats that as a rejection and moves on. The actual LLM call is delegated to an
agent (agents.SingleCoder / OptimizerTeam) via its complete() interface, so the
single-vs-team ablation never touches this file.
"""
import re

from ..artifact.program import SearchProgram


def extract_code(text):
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no fenced code block in LLM response")
    return blocks[-1].strip() + "\n"


def _format_failures(failures):
    if not failures:
        return "(none -- every rollout query had perfect recall@20)"
    parts = []
    for i, f in enumerate(failures, 1):
        sec = [f"[{i}] query: {f['query']}"]
        sec.append(f"    recall@20={f['recall@20']:.2f} hit@1={f['hit@1']:.0f} "
                   f"mrr={f['mrr']:.2f}  gold ids: {f['gold_ids']}")
        for nid, text in f["missed"]:
            sec.append(f"    MISSED gold {nid}: {text}")
        sec.append(f"    retrieved top-20 ids: {f['retrieved']}")
        if f.get("error"):
            sec.append(f"    ERROR: {f['error']}")
        parts.append("\n".join(sec))
    return "\n".join(parts)


def _format_buffer(buffer_entries):
    if not buffer_entries:
        return "(none yet)"
    parts = []
    for e in buffer_entries:
        after = e.get("score_after")
        after_s = f"{after['recall@20']:.4f}" if after else "n/a (failed before gate)"
        parts.append(
            f"- step {e['step']}: recall@20 {e['score_before']['recall@20']:.4f} -> "
            f"{after_s}; reason: {e['reason']}\n{e['diff']}")
    return "\n".join(parts)


def build_prompt(incumbent_src, primitives_doc, failures, buffer_entries,
                 max_edits, safe_builtins, momentum=""):
    momentum_block = f"\n## Search momentum (what has been working)\n{momentum}\n" if momentum else ""
    return f"""\
You are optimizing a Python retrieval program for STaRK-prime, a biomedical \
knowledge-graph QA benchmark. Given a natural-language query, the program \
must return candidate node ids ranked by relevance. The metric being \
optimized is recall@20 (secondary: hit@1, MRR), averaged over held-out \
queries. Gold answer sets often contain MANY nodes (median ~10), and answers \
can be of any node type.

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

## Worst failures from the latest rollout (train queries the incumbent missed)
{_format_failures(failures)}

## Previously rejected edits (these did NOT improve val recall@20 -- do not repeat them)
{_format_buffer(buffer_entries)}{momentum_block}

## Your task
1. Diagnose in a few sentences WHY the incumbent misses these gold nodes
   (look at what the missed nodes' texts share, and which primitives could
   reach them).
2. Propose ONE improved program. Edit budget: at most {max_edits} changed
   regions (difflib hunks) vs the incumbent -- prefer small, targeted edits
   over rewrites. For each edit give a one-line rationale.
3. End your reply with the COMPLETE new module in a single ```python fenced
   block (it replaces the whole file; it must compile under the contract).
"""


class Mutator:
    def __init__(self, agent, primitives_doc, safe_builtins):
        self._agent = agent
        self._doc = primitives_doc
        self._safe_builtins = safe_builtins

    def propose(self, program: SearchProgram, failures, buffer_entries,
                edit_budget, momentum=""):
        """-> (candidate SearchProgram or None, transcript:str)."""
        prompt = build_prompt(program.src, self._doc, failures, buffer_entries,
                              edit_budget, self._safe_builtins, momentum)
        transcript = [f"# PROMPT\n\n{prompt}"]
        for attempt in (1, 2):
            try:
                resp = self._agent.complete(prompt)
            except Exception as e:
                transcript.append(f"# AGENT CALL FAILED (attempt {attempt})\n\n{e}")
                return None, "\n\n---\n\n".join(transcript)
            transcript.append(f"# RESPONSE (attempt {attempt})\n\n{resp}")
            try:
                cand_src = extract_code(resp)
            except ValueError as e:
                prompt += (f"\n\nYour previous reply had no fenced code block "
                           f"({e}). Reply again, ending with the complete module "
                           "in ONE ```python block.")
                continue
            cand = program.with_src(cand_src)
            hunks = program.edit_distance(cand)
            if hunks <= edit_budget:
                return cand, "\n\n---\n\n".join(transcript)
            prompt += (f"\n\nYour previous candidate changed {hunks} regions but "
                       f"the budget is {edit_budget}. Keep your best {edit_budget} "
                       "edits, revert the rest, and reply again with the complete "
                       "module in ONE ```python block.")
        transcript.append("# SKIPPED: edit budget exceeded twice")
        return None, "\n\n---\n\n".join(transcript)
