"""primitives.py -- the closed DSL contract: argument validation, caps, and
allowlist checks shared by every RetrievalGraph primitive.

This module holds NO Cypher and no connection -- it is the parameterization
wall. RetrievalGraph composes these helpers with the backend to emit the
actual (parameterized, capped, read-only) queries. Keeping them separate is
what lets the candidate program call G.* freely while every argument that
could reach the query string is validated here first.
"""
import numpy as np

MAX_IDS_IN = 1000  # hard cap on caller-supplied id-lists


def as_int(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an int, got {type(x).__name__}")
    return int(x)


def clamp(x, name, hi, lo=1):
    return max(lo, min(as_int(x, name), hi))


def ids_in(ids, cap=MAX_IDS_IN):
    """Normalize an int or iterable-of-ints to a capped list[int]."""
    if isinstance(ids, (int, np.integer)) and not isinstance(ids, bool):
        ids = [ids]
    try:
        ids = list(ids)
    except TypeError:
        raise ValueError("ids must be an int or an iterable of ints")
    if not ids:
        raise ValueError("ids is empty")
    return [as_int(i, "ids[*]") for i in ids[:cap]]


def nonempty_str(s, name):
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return s


class Allowlists:
    """Labels / relationship-types / node-types read from the live graph at
    startup. A value not on the list raises -- this is how rel-type and label
    args (which cannot be query parameters) are kept injection-free."""

    def __init__(self, labels, rel_types, ntypes):
        self.labels = tuple(labels)                 # vector-indexed per-type labels
        self.rel_types = tuple(rel_types)
        self.ntypes = tuple(ntypes)

    def check_label(self, label):
        if label not in self.labels:
            raise ValueError(f"unknown label {label!r}; valid: {list(self.labels)}")
        return label

    def check_rel(self, rel_type):
        if rel_type is None:
            return None
        if rel_type not in self.rel_types:
            raise ValueError(f"unknown rel_type {rel_type!r}; valid: {list(self.rel_types)}")
        return rel_type

    def check_ntype(self, ntype):
        if ntype not in self.ntypes:
            raise ValueError(f"unknown ntype {ntype!r}; valid: {list(self.ntypes)}")
        return ntype
