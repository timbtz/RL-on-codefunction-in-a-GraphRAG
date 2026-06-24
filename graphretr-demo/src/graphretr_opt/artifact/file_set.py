"""FileSet -- the multi-file layer-2 artifact (the `graph_search` target).

Where SearchProgram (artifact/program.py) carries a single `def search(q, G)`
module as source text, FileSet carries an *overlay* of edited files on top of an
immutable base checkout (the in-repo `graphsearch/src`). The product of the
optimizer on this path is this object's `overlay` evolving over a diff history.

This is the "whole-file artifact" of kernel-evo: the thing being optimized is
real source files, not a re-expression inside an AST jail. The base lives on
disk and is never mutated; an edit produces a sibling FileSet with a new
overlay. `materialize(dest)` writes base+overlay into a throwaway directory the
subprocess eval seam (env/targets/subprocess_target.py) imports from.

Mirrors SearchProgram's surface (sha / with_* / edit_distance / diff) so the
mutator, edit-budget, pool and checkpoint code stay artifact-agnostic -- only
the apply step differs (apply_edits_multi vs apply_edits). SearchProgram is left
intact; FileSet is a sibling, not a replacement.
"""
import difflib
import hashlib
import os
import shutil


class FileSet:
    def __init__(self, base_dir, overlay, editable, family="graph_search"):
        """base_dir: the immutable base checkout (graphsearch/src).
        overlay: {relpath -> source} -- the edited files (relpath is relative to
                 base_dir, e.g. "common/service/search/...py").
        editable: tuple of relpaths the mutator is allowed to touch.
        """
        self.base_dir = base_dir
        # normalize separators so sha/equality are stable across platforms
        self.overlay = {self._norm(k): v for k, v in dict(overlay).items()}
        self.editable = tuple(self._norm(p) for p in editable)
        self.family = family

    @staticmethod
    def _norm(relpath):
        return relpath.replace(os.sep, "/").lstrip("/")

    @classmethod
    def from_base(cls, base_dir, editable, family="graph_search"):
        """Seed a FileSet whose overlay is the *current, unmodified* source of
        every editable file read straight off base_dir (the run starts from the
        real service code)."""
        overlay = {}
        for rel in editable:
            rel = cls._norm(rel)
            with open(os.path.join(base_dir, rel)) as f:
                overlay[rel] = f.read()
        return cls(base_dir, overlay, editable, family=family)

    @property
    def sha(self) -> str:
        """Content hash of the overlay (sorted by relpath -- order-independent).
        The base_dir is excluded: two FileSets with the same edits score the
        same regardless of where the base checkout lives."""
        h = hashlib.sha256()
        for rel in sorted(self.overlay):
            h.update(rel.encode())
            h.update(b"\0")
            h.update(self.overlay[rel].encode())
            h.update(b"\0")
        return h.hexdigest()

    @property
    def src(self):
        """Pass-through token so FileSet flows through FastLoop's SearchProgram-
        shaped call sites (`compile(p.src)`, `score(..., src=p.src)`) unchanged:
        NullSandbox.compile returns it as-is and McqRewardAdapter ignores the
        `src=` kwarg (it scores the FileSet it is handed). The artifact is NEVER
        serialized via `.src` -- pool/checkpoint use to_dict() instead."""
        return self

    @property
    def primary_file(self):
        """The first editable relpath present in the overlay (the headline file
        accepted candidates are saved as)."""
        for rel in self.editable:
            if rel in self.overlay:
                return rel
        return next(iter(self.overlay), None)

    def src_of(self, relpath) -> str:
        """The current source of one overlay file."""
        return self.overlay[self._norm(relpath)]

    def with_overlay(self, new: dict) -> "FileSet":
        """A sibling FileSet (same base/editable/family) whose overlay is
        REPLACED by `new` -- the multi-file 'apply_edits'."""
        return FileSet(self.base_dir, new, self.editable, family=self.family)

    def edit_distance(self, other: "FileSet") -> int:
        """Number of non-equal difflib hunks across all changed files == the edit
        budget L the mutator is held to (sum over the union of relpaths)."""
        total = 0
        for rel in set(self.overlay) | set(other.overlay):
            a = self.overlay.get(rel, "").splitlines()
            b = other.overlay.get(rel, "").splitlines()
            sm = difflib.SequenceMatcher(None, a, b)
            total += sum(1 for op, *_ in sm.get_opcodes() if op != "equal")
        return total

    def change_summary(self, other: "FileSet", max_changes: int = 2) -> str:
        """One-line gist of the edit vs another FileSet (for the rejected
        buffer) -- the first few changed lines across files."""
        changed = []
        for rel in sorted(set(self.overlay) | set(other.overlay)):
            for l in difflib.unified_diff(
                    self.overlay.get(rel, "").splitlines(),
                    other.overlay.get(rel, "").splitlines(), lineterm=""):
                if l[:1] in "+-" and not l.startswith(("+++", "---")):
                    changed.append(l)
        head = "  ".join(l.strip() for l in changed[:max_changes * 2])
        extra = len(changed) - max_changes * 2
        return (head + (f" (+{extra} more lines)" if extra > 0 else "")
                or "(no textual change)")

    def diff(self, other: "FileSet", max_lines: int = 80) -> str:
        """Unified diff across every changed file, file headers included."""
        lines = []
        for rel in sorted(set(self.overlay) | set(other.overlay)):
            a = self.overlay.get(rel, "").splitlines()
            b = other.overlay.get(rel, "").splitlines()
            if a == b:
                continue
            lines += list(difflib.unified_diff(
                a, b, fromfile=f"incumbent/{rel}", tofile=f"candidate/{rel}",
                lineterm=""))
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... (+{len(lines) - max_lines} more diff lines)"]
        return "\n".join(lines)

    def materialize(self, dest_dir):
        """Copy base_dir -> dest_dir, then overwrite each overlay file. The
        resulting tree is a complete, importable copy of graphsearch/src with the
        candidate edits applied -- what the subprocess worker puts on sys.path.
        dest_dir must NOT already exist (copytree creates it)."""
        shutil.copytree(self.base_dir, dest_dir)
        for rel, src in self.overlay.items():
            path = os.path.join(dest_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(src)
        return dest_dir

    def save(self, path):
        """Persist the candidate. FastLoop saves accepted artifacts to a `.py`
        path (e.g. accepted/step_003.py); for a FileSet that means the PRIMARY
        editable file's source (the headline diffable artifact), with the full
        overlay tree written alongside under a sibling `<name>.files/` dir so no
        edited file is lost. A non-`.py` path is treated as a directory and gets
        the overlay tree directly."""
        if path.endswith(".py"):
            primary = self.primary_file
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                f.write(self.overlay.get(primary, "") if primary else "")
            if len(self.overlay) > 1:
                self._save_tree(path[:-3] + ".files")
            return
        self._save_tree(path)

    def _save_tree(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        for rel, src in self.overlay.items():
            p = os.path.join(dir_path, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(src)

    def to_dict(self) -> dict:
        """Checkpoint payload -- base_dir + overlay + editable (JSON-safe)."""
        return {"base_dir": self.base_dir, "overlay": dict(self.overlay),
                "editable": list(self.editable), "family": self.family}

    @classmethod
    def from_dict(cls, d: dict) -> "FileSet":
        return cls(d["base_dir"], d["overlay"], d["editable"],
                   family=d.get("family", "graph_search"))
