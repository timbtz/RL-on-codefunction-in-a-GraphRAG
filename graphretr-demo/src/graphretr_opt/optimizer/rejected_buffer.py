"""RejectedBuffer -- negative replay. Every rejected candidate is remembered as
a ONE-LINE summary carrying the edit gist and the score delta it caused (e.g.
"k=60 -> k=100 => recall@20 -0.04; gate not improved"); the last-N enter the
next reflection prompt so the mutator stops re-proposing dead ends.

Storing a summary (not the full unified diff) is deliberate: in run 3 the raw
diff dump was ~60% of the prompt and buried the actual failure signal. The
quantified delta steers better than a bare diff anyway (SkillOpt finding).

Stage-1: a flat ring (no epochs); the full buffer is persisted as an artifact,
only the last `context_n` entries enter the prompt.
"""
import json

from ..atomic_io import atomic_write_json


class RejectedBuffer:
    def __init__(self, context_n=8):
        self.entries = []
        self.context_n = context_n

    def add(self, step, summary, outcome="reject"):
        self.entries.append({"step": step, "summary": summary, "outcome": outcome})

    def _recent(self, outcome, n):
        """Most-recent-first then re-chronological, DEDUPED by normalized summary,
        the last `n` entries of the given outcome. Dedup is the mode-collapse fix
        (run7/8): a dead OR winning idea is shown once, and old distinct ideas are
        not crowded out by the proposer re-submitting the same edit every step."""
        seen, out = set(), []
        for e in reversed(self.entries):
            if e.get("outcome", "reject") != outcome:
                continue
            key = " ".join(str(e.get("summary", "")).split()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
            if len(out) >= n:
                break
        return list(reversed(out))

    def recent(self):
        """Deduped recent REJECTED edits -- negative replay ('do not repeat')."""
        return self._recent("reject", self.context_n)

    def accepted(self):
        """Deduped recent ACCEPTED edits -- positive replay ('build on these')."""
        return self._recent("accept", self.context_n)

    def load(self, path):
        try:
            self.entries = json.load(open(path))
        except FileNotFoundError:
            pass
        return self

    def save(self, path):
        atomic_write_json(path, self.entries, indent=1)
