"""RejectedBuffer -- negative replay. Every rejected candidate's diff + score
drop is remembered; the last-N are rendered into the next reflection prompt so
the mutator stops re-proposing dead ends.

Stage-1: a flat ring (no epochs). The full buffer is persisted as an artifact;
only the last `context_n` entries enter the prompt.
"""
import json


class RejectedBuffer:
    def __init__(self, context_n=8):
        self.entries = []
        self.context_n = context_n

    def add(self, step, diff, score_before, score_after, reason):
        self.entries.append({
            "step": step,
            "reason": reason,
            "score_before": score_before,   # plain dict, JSON-safe
            "score_after": score_after,     # dict or None
            "diff": diff,
        })

    def recent(self):
        return self.entries[-self.context_n:]

    def load(self, path):
        try:
            self.entries = json.load(open(path))
        except FileNotFoundError:
            pass
        return self

    def save(self, path):
        json.dump(self.entries, open(path, "w"), indent=1)
