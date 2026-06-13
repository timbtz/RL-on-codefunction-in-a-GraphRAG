"""TieredCoder -- model tiering behind the SingleCoder interface.

The user's mental model made real: a cheap **analyst** (Haiku) compresses the
rollout evidence into a tight briefing, a mid-tier **editor** (Sonnet) proposes
the edit, and the loop escalates to the **architect** (Opus) only when stuck
(plateau). It exposes the exact digest()/complete(prompt, tier) seam SingleCoder
does, so Mutator and FastLoop never branch on which agent is wired -- the
single-vs-tiered comparison is a drop-in swap under an equal token budget.

Token economics vs the old all-Opus path: the analyst pre-pass is a few hundred
Haiku tokens that shrink the (expensive) editor prompt, and routine edits run on
Sonnet instead of Opus. Per-tier call counts are logged so the cost is auditable.
"""
from collections import Counter

from .single import SingleCoder

_ANALYST = """You are the analyst in a code-optimization loop. Compress the \
evidence below into a TIGHT briefing (3-6 bullets) for the engineer who will \
edit a graph-retrieval program. Cover: what the program is missing or \
mis-ranking, WHY (what the missed gold texts and the out-ranking non-gold texts \
have in common), which graph primitives could reach or re-rank those nodes, and \
which already-rejected edits NOT to repeat. No preamble, no code, no restating \
the program. Evidence:

"""


class TieredCoder:
    def __init__(self, backend="cli", analyst="haiku", editor="sonnet",
                 architect="opus", timeout_s=900):
        self.analyst = SingleCoder(backend, analyst, timeout_s)
        self.editor = SingleCoder(backend, editor, timeout_s)
        self.architect = SingleCoder(backend, architect, timeout_s)

    @property
    def call_counts(self) -> Counter:
        c = Counter()
        for sub in (self.analyst, self.editor, self.architect):
            c.update(sub.call_counts)
        return c

    def digest(self, evidence: str) -> str:
        return self.analyst.complete(_ANALYST + evidence)

    def complete(self, prompt: str, tier="editor") -> str:
        coder = self.architect if tier == "architect" else self.editor
        return coder.complete(prompt)
