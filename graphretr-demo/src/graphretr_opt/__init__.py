"""graphretr_opt -- target-agnostic optimizer ENGINE that rewrites a search
program/service over a frozen graph environment (LLM frozen, artifact = code).

The engine drives two sibling search SERVICES (carved out 2026-06-25): the STaRK
service in `starksearch/` and the company-KG service in `graphsearch/`. What
remains here is the machinery both reuse:
  optimizer/  the "RL" half: proposes bounded edits, scores against the target,
              gates, archives. SkillOpt mechanisms live here.
  env/        the eval seam: the SearchTarget port + subprocess targets, the
              shared LLM/budget gateway, the stable SandboxError, and (until the
              STaRK path goes subprocess) the in-process AST Sandbox.
  reward/     the scoring FRAMEWORK (MetricVector, pareto dominance) -- per-target
              scorers live with their service.

The artifact that flows through the loop is a SearchProgram (single-file) or a
FileSet (multi-file service overlay). Stage 1 drives optimizer/fast_loop.py via
campaign.py/cli.py; the cross-strategy slow-loop layer is deferred (its inert
stubs were removed 2026-06-25 -- recoverable from git when Stage 2 ships).
"""
