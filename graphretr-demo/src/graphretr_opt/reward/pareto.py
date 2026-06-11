"""pareto.py -- dominance test, frontier maintenance, and MAP-Elites cells.

Objectives (all "higher is better" after sign-flipping costs):
  + quality recall@20   (maximize)
  - latency_s           (minimize -> negate)
  - db_load             (minimize -> negate)
  - code_complexity     (minimize -> negate)

Stage 1 uses this only as a passive record alongside the strict-greater gate
(so the demo can SHOW the frontier even though acceptance is single-axis).
Stage 2's SlowLoop and GEPA select on it: dominance for the gate, cells for
niching.
"""
from dataclasses import dataclass, field


def _objective_tuple(mv):
    return (mv.primary, -mv.latency_s, -mv.db_load, -mv.code_complexity)


def dominates(a, b) -> bool:
    """a dominates b: >= on every objective and > on at least one."""
    ta, tb = _objective_tuple(a), _objective_tuple(b)
    return all(x >= y for x, y in zip(ta, tb)) and any(x > y for x, y in zip(ta, tb))


@dataclass
class Entry:
    sha: str
    family: str
    metrics: object  # MetricVector


@dataclass
class ParetoArchive:
    entries: list = field(default_factory=list)
    cells: dict = field(default_factory=dict)  # MAP-Elites: cell -> best Entry

    def add(self, sha, family, metrics) -> bool:
        """Insert if non-dominated; drop entries it dominates. Returns True if
        it made (or updated) the frontier."""
        if any(dominates(e.metrics, metrics) for e in self.entries):
            kept = False
        else:
            self.entries = [e for e in self.entries if not dominates(metrics, e.metrics)]
            self.entries.append(Entry(sha, family, metrics))
            kept = True
        # MAP-Elites niche: (quality bucket, cost bucket); keep best-quality per cell.
        c = self.cell(metrics)
        cur = self.cells.get(c)
        if cur is None or metrics.primary > cur.metrics.primary:
            self.cells[c] = Entry(sha, family, metrics)
        return kept

    @staticmethod
    def cell(metrics, q_bins=20, cost_bins=10):
        """(quality, -cost) niche coordinate for MAP-Elites."""
        q = min(q_bins - 1, int(metrics.primary * q_bins))
        cost = min(cost_bins - 1, int(metrics.db_load))
        return (q, cost)

    def best(self, metric_attr="primary"):
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: getattr(e.metrics, metric_attr))
