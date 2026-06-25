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

propose() returns (candidate SearchProgram or None, transcript, meta). None
means the step is skipped (budget blown, unparseable reply, transient agent
failure, or -- with repair on -- a candidate the sandbox/probe kept rejecting).
`meta` carries the reject reason and the probe-failed count so the loop can set
its step `reason` precisely. AgentUnavailable (hard CLI/spend limit) propagates
so the loop can stop the campaign instead of burning the remaining steps.

Compiler-in-the-loop self-repair (KernelEvolve read/replace/lint): when the
caller supplies a `validate(cand)` closure (it compiles + probes, raising
SandboxError on reject) and a `repair_budget > 0`, a candidate that parses and
fits the edit budget but FAILS validation is fed its exact sandbox/probe error
back into the SAME conversation for a targeted fix -- cheaper than today's
reject-and-repropose-next-step because it reuses the already-computed
digest + prompt. Default `repair_budget=0` => validation never runs and the old
behavior (caller compiles/probes after propose returns) is byte-identical.
"""
from ..agents.single import AgentUnavailable
from ..artifact.program import SearchProgram
from ..env.errors import SandboxError
from .edits import (EditError, apply_edits, apply_edits_multi, extract_code,
                    parse_edit_blocks, parse_edit_blocks_multi)


def _is_fileset(program):
    """Duck-type the artifact kind: a FileSet carries an `overlay` dict, a
    SearchProgram carries a single `src`. Keeping the switch structural (not an
    isinstance on a concrete class) mirrors how the loop duck-types Substrate /
    RewardModel -- the mutator never imports FileSet."""
    return hasattr(program, "overlay")


def _fmt_metrics(m):
    return (f"recall@20={m['recall@20']:.2f} hit@1={m['hit@1']:.0f} "
            f"hit@5={m.get('hit@5', 0.0):.0f} mrr={m['mrr']:.2f}")


def _format_failures(failures):
    if not failures:
        return "(none -- every rollout query had perfect recall@20 and hit@1)"
    parts = []
    for i, f in enumerate(failures, 1):
        sec = [f"[{i}] ({f['bucket']}) {f['query']}", f"    {_fmt_metrics(f['metrics'])}"]
        if f.get("attribution"):
            sec.append(f"    -> {f['attribution']}")
        if f.get("empty_result"):
            sec.append("    EMPTY: search() returned no candidates for this query "
                       "(dead traversal -- broaden retrieval before re-ranking)")
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


def format_evidence(failures, wins, buffer_entries, accepted_entries=None):
    """The rollout evidence block -- what agent.digest() compresses."""
    won = ""
    if accepted_entries:
        won = ("## Edits that IMPROVED the metric so far -- build on these, do not undo them\n"
               + _format_buffer(accepted_entries) + "\n\n")
    return f"""## Worst failures from the latest rollout (train queries)
{_format_failures(failures)}

{_format_wins(wins)}{won}## Previously rejected edits (did NOT improve the gate -- do not repeat)
{_format_buffer(buffer_entries)}"""


def build_prompt(incumbent_src, primitives_doc, digest, max_edits, safe_builtins,
                 plateau=False):
    stance = (
        "You have STALLED for several steps -- small tweaks are not working. Make "
        "a BOLD, STRUCTURAL change this step: rewrite an operator end-to-end, "
        "replace the retrieval strategy, or add a new graph/hybrid recall path. Do "
        "NOT just retune constants or thresholds. A wholesale rewrite (end with the "
        "COMPLETE module in one ```python block) is encouraged here."
        if plateau else
        "Prefer focused edits, but rewrite a whole operator -- its Cypher or its "
        "LLM prompt -- whenever that is what moves the metric. You are NOT limited "
        "to small tweaks.")
    return f"""\
You are optimizing a Python retrieval program for STaRK-prime, a biomedical \
knowledge-graph QA benchmark. Given a natural-language query, the program must \
return candidate node ids ranked by relevance. The gate optimizes a blend of \
recall@20 (gold anywhere in top-20) AND ranking quality (hit@1 / MRR). Gold \
answer sets often contain MANY nodes (median ~10), of any node type.

You OWN THE ENTIRE PIPELINE. The retrieval, extraction and re-ranking operators \
are ordinary helper functions in the program below, built on three raw primitives \
(G.query read-only Cypher, G.embed, G.llm). You may rewrite their Cypher, rewrite \
their LLM prompts, change the fusion math, or add brand-new operators. BOTH \
failure modes are in scope: if gold is NOT in the pool at all (GENERATION/recall \
failure), write better or different retrieval -- hybrid text+vector fusion, or a \
GRAPH walk (k-hop / shortest_path between the query's entities via G.query) that \
reaches gold the embeddings miss; if gold IS in the pool but ranked low (RANKING \
failure), fix the scoring / rerank prompt. Do NOT assume the stock embedding \
retrieval is good enough -- it often is not.

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
   operators/queries could reach or re-rank them). ATTRIBUTE each failure using
   its `->` tag: a RANKING failure (gold in top-100 but not top-20) needs a
   SCORING/rerank fix; a GENERATION failure (gold NOT in top-100) needs broader,
   reformulated, or GRAPH-STRUCTURAL retrieval -- rewrite the retrieval operator
   or add a G.query graph walk, do not just re-rank. Name the specific
   helper/call responsible.
2. Propose your change as SEARCH/REPLACE blocks (up to {max_edits}). Each block:
   <<<<<<< SEARCH
   <exact contiguous lines copied verbatim from the incumbent>
   =======
   <the replacement lines>
   >>>>>>> REPLACE
   The SEARCH text must match the incumbent EXACTLY and appear exactly once.
   {stance}
   Give a one-line rationale per block. For a wholesale rewrite, end your reply
   with the COMPLETE module in ONE ```python block instead.
"""


def build_search_prompt(file_set, primitives_doc, digest, max_edits):
    """Edit prompt for the `graph_search` target -- the real agentic search
    service file(s), not a sandboxed `def search(q, G)`. Renders each editable
    file under a `FILE:` header (the same header the multi-file applier reads
    back) so the model edits whole real source. No AST execution contract: the
    only contract is "the edited service still imports and `search(query) -> str`
    returns retrieved context" -- enforced by the subprocess eval seam, not a
    gate. `primitives_doc` here is a short domain note (Neo4j schema / cost
    levers), not a `G.*` API."""
    files_block = "\n\n".join(
        f"FILE: {rel}\n```python\n{file_set.overlay[rel]}```"
        for rel in file_set.editable if rel in file_set.overlay)
    return f"""\
You are optimizing a REAL agentic graph-retrieval service for a German company \
knowledge graph (Neo4j fulltext + recursive traversal, no embeddings). Given a \
natural-language query, `search(query) -> str` returns a retrieved-context \
string with [doc:ID] citations. A separate, FROZEN answerer then picks a \
multiple-choice answer from ONLY that context. You are scored on how often the \
answerer is then correct (closed-book-adjusted) plus whether the gold source \
document appears in your context -- minus cost (LLM calls / latency). So: surface \
the RIGHT documents into the context; do not try to answer the question yourself.

## Domain notes
{primitives_doc}

## Editable file(s) (the incumbent)
{files_block}

## How your change is evaluated
- The edited file replaces the original inside a throwaway copy of the service,
  which is imported and run over a gold Q&A set in an isolated subprocess.
- A file that fails to import, crashes, or hangs scores as a crash for those
  queries (the subprocess is killed on a wall-clock timeout). Prefer changes that
  keep the service importable and the `search` signature intact.
- Cheaper retrieval that keeps quality is rewarded (fewer LLM calls / lower
  latency are cost axes).

## Evidence (latest rollout -- per-question retrieval/answer outcomes)
{digest}

## Your task
1. Diagnose in a few sentences WHY the answerer misses on these questions. Use the
   per-question signal: a RETRIEVAL miss (the gold source is absent from the
   returned context) needs broader/deeper traversal or better keyword extraction;
   a REASONING miss (gold source WAS in context but the answer was still wrong)
   usually means too much noise -- tighten relevance filtering or compression.
2. Propose your change as up to {max_edits} SEARCH/REPLACE blocks. Each block:
   FILE: <one of the editable relpaths above>
   <<<<<<< SEARCH
   <exact contiguous lines copied verbatim from that file>
   =======
   <the replacement lines>
   >>>>>>> REPLACE
   The SEARCH text must match that file EXACTLY and appear exactly once. The
   FILE: header is required when more than one file is editable. Prefer small,
   targeted edits; give a one-line rationale per block.
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

    def _candidate_multi(self, file_set, resp):
        """FileSet variant of `_candidate`. Apply FILE:-tagged SEARCH/REPLACE
        blocks to the overlay. -> (FileSet, n_edits) or (None, 0). There is no
        full-module fallback here: a whole-file rewrite is ambiguous across
        several files, so an unparseable/unanchorable reply just skips the step
        (the loop re-prompts via the base-attempt retry)."""
        blocks = parse_edit_blocks_multi(resp)
        if not blocks:
            return None, 0
        try:
            new_overlay = apply_edits_multi(
                file_set.overlay, blocks, editable=file_set.editable)
        except EditError:
            return None, 0
        return file_set.with_overlay(new_overlay), len(blocks)

    def propose(self, program: SearchProgram, failures, wins, buffer_entries,
                edit_budget, plateau=False, validate=None, repair_budget=0,
                accepted_entries=None):
        """Propose an edit to `program` from the rollout evidence.

        -> (candidate SearchProgram or None, transcript:str, meta:dict) where
        meta = {"reject_reason": "parse"|"budget"|"sandbox"|"agent"|None,
                "probe_failed": int}. reject_reason is None on success.

        Two independent retry pools share the one conversation:
          * parse/budget retries consume the 2 BASE attempts (a reply with no
            usable block, or one over the edit budget);
          * sandbox/probe retries consume `repair_budget` -- only reached when
            `validate(cand)` is supplied. validate compiles + probes the
            candidate and raises SandboxError on reject; its exact text is fed
            back for a targeted fix (compiler-in-the-loop self-repair).
        With validate=None / repair_budget=0 the validation arm never runs."""
        evidence = format_evidence(failures, wins, buffer_entries, accepted_entries)
        transcript = [f"# EVIDENCE\n\n{evidence}"]
        meta = {"reject_reason": None, "probe_failed": 0}

        def _done(cand, reason):
            meta["reject_reason"] = reason
            return cand, "\n\n---\n\n".join(transcript), meta

        try:
            digest = self._agent.digest(evidence)
        except AgentUnavailable:
            raise
        except Exception as e:           # analyst hiccup -> use raw evidence
            transcript.append(f"# ANALYST FAILED -- using raw evidence\n\n{e}")
            digest = evidence
        if digest is not evidence:
            transcript.append(f"# DIGEST (analyst)\n\n{digest}")

        is_fs = _is_fileset(program)
        # Bold mode on plateau: lift the edit budget so a wholesale restructure
        # is not bounced as "over budget" (a full-module rewrite has a large edit
        # distance), and tell the proposer to make a structural change.
        eff_budget = 9999 if plateau else edit_budget
        if is_fs:
            prompt = build_search_prompt(program, self._doc, digest, eff_budget)
        else:
            prompt = build_prompt(program.src, self._doc, digest, eff_budget,
                                  self._safe_builtins, plateau=plateau)
        transcript.append(f"# PROMPT\n\n{prompt}")
        tier = "architect" if plateau else "editor"
        base_left = 2                       # parse + over-budget retries
        repairs_left = max(0, int(repair_budget))  # sandbox/probe repairs
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._agent.complete(prompt, tier=tier)
            except AgentUnavailable:
                raise
            except Exception as e:
                transcript.append(f"# AGENT CALL FAILED (attempt {attempt})\n\n{e}")
                return _done(None, "agent")
            transcript.append(f"# RESPONSE ({tier}, attempt {attempt})\n\n{resp}")

            cand, n_edits = (self._candidate_multi(program, resp) if is_fs
                             else self._candidate(program, resp))
            if cand is None:
                base_left -= 1
                if base_left <= 0:
                    transcript.append("# SKIPPED: no usable candidate after "
                                      "2 base attempts")
                    return _done(None, "parse")
                prompt += ("\n\nYour previous reply contained neither a usable "
                           "SEARCH/REPLACE block (the SEARCH text must match the "
                           "incumbent exactly) nor a ```python module. Reply again.")
                continue
            if n_edits > eff_budget:
                base_left -= 1
                if base_left <= 0:
                    transcript.append("# SKIPPED: no in-budget candidate after "
                                      "2 base attempts")
                    return _done(None, "budget")
                prompt += (f"\n\nYour previous candidate changed {n_edits} regions "
                           f"but the budget is {eff_budget}. Keep your best "
                           f"{eff_budget} edits and reply again.")
                continue
            # Parsed and in budget. Validate (compile + probe) when asked, and
            # self-repair the exact sandbox/probe error in this same conversation.
            if validate is not None:
                try:
                    validate(cand)
                except SandboxError as e:
                    meta["probe_failed"] += 1
                    if repairs_left <= 0:
                        transcript.append(f"# REJECTED BY SANDBOX/PROBE "
                                          f"(repair budget exhausted)\n\n{e}")
                        return _done(None, "sandbox")
                    repairs_left -= 1
                    transcript.append(f"# REJECTED BY SANDBOX/PROBE "
                                      f"(repairs left {repairs_left})\n\n{e}")
                    prompt += (
                        "\n\nYour previous candidate was REJECTED by the "
                        f"sandbox/probe with this exact error:\n  {e}\n"
                        "It compiled but failed validation. Fix THIS error "
                        "specifically and reply again.")
                    continue
            return _done(cand, None)
