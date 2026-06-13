"""SearchProgram -- the mutable layer-2 artifact (SkillOpt's `s`).

It is a `def search(q, G)` module as source text plus a strategy-family tag.
The whole product of the optimizer is this object's `.src` evolving over a diff
history. The program is executed inside env.Sandbox, never here -- this class
only carries the source, hashes/diffs it, and counts edit hunks.

Stage-1 mutation replaces `.src` wholesale (bounded by diff-hunk budget, which
is what apply_edits/edit_distance count). Stage-2 can add structured Edit
objects (add/delete/replace spans); the call sites already speak in terms of
"a new SearchProgram with N hunks changed", so that refinement is local.
"""
import difflib
import hashlib


class SearchProgram:
    def __init__(self, src: str, family: str = "unknown"):
        self.src = src
        self.family = family

    @classmethod
    def from_file(cls, path: str, family: str = "unknown"):
        return cls(open(path).read(), family=family)

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.src.encode()).hexdigest()

    def with_src(self, new_src: str) -> "SearchProgram":
        """A sibling program (same family) carrying replacement source -- the
        Stage-1 'apply_edits': the mutator returns a full new body."""
        return SearchProgram(new_src, family=self.family)

    def edit_distance(self, other: "SearchProgram") -> int:
        """Number of non-equal difflib hunks vs another program == the edit
        budget L the mutator is held to."""
        sm = difflib.SequenceMatcher(None, self.src.splitlines(),
                                     other.src.splitlines())
        return sum(1 for op, *_ in sm.get_opcodes() if op != "equal")

    def change_summary(self, other: "SearchProgram", max_changes: int = 2) -> str:
        """One-line gist of the edit vs another program -- the first few changed
        lines, for the rejected buffer (the full diff is too big to keep)."""
        changed = [l for l in difflib.unified_diff(
            self.src.splitlines(), other.src.splitlines(), lineterm="")
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        head = "  ".join(l.strip() for l in changed[:max_changes * 2])
        extra = len(changed) - max_changes * 2
        return (head + (f" (+{extra} more lines)" if extra > 0 else "")
                or "(no textual change)")

    def diff(self, other: "SearchProgram", max_lines: int = 80) -> str:
        lines = list(difflib.unified_diff(
            self.src.splitlines(), other.src.splitlines(),
            fromfile="incumbent", tofile="candidate", lineterm=""))
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... (+{len(lines) - max_lines} more diff lines)"]
        return "\n".join(lines)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.src)
