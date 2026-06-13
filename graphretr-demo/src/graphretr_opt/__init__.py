"""graphretr_opt -- optimizer service that rewrites a search program over a
frozen graph environment (LLM frozen, artifact = code).

Two long-lived halves with a hard wall between them:
  env/        the immutable environment: graph backend + closed primitive DSL
              + sandbox. Never mutated by the loop.
  optimizer/  the "RL" half: proposes bounded edits, scores against the env,
              gates, archives. SkillOpt mechanisms live here.

The artifact that flows between them is artifact/program.py:SearchProgram
(layer 2 of the three-layer model) -- never the primitives (layer 1) or the
strategy family (layer 3, slow loop).

Stage 1 (this demo) drives optimizer/fast_loop.py via campaign.py/cli.py and
leaves slow_loop, scheduler and gepa_adapter as documented seams for the
deferred cross-strategy exploration layer.
"""
