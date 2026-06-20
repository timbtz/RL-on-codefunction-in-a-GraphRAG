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

    def add(self, step, summary):
        self.entries.append({"step": step, "summary": summary})

    def recent(self):
        return self.entries[-self.context_n:]

    def load(self, path):
        try:
            self.entries = json.load(open(path))
        except FileNotFoundError:
            pass
        return self

    def save(self, path):
        atomic_write_json(path, self.entries, indent=1)
