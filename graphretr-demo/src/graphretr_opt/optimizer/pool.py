"""CandidatePool -- the instance-wise Pareto pool that replaces run-5's single
incumbent (Phase A, GEPA Alg. 2). The old `ParetoArchive` kept only the
aggregate frontier as a passive record (never read, since removed); this pool
actively drives parent SELECTION and the stop condition.

A child is ADMITTED to the pool if it is either
  * non-dominated on the aggregate metric vector (a Pareto-frontier point), OR
  * sole-best on >= 1 individual gate query (instance-wise) -- a specialist the
    aggregate would discard but that carries a unique skill worth recombining.

Parent SELECTION (each step): among members that win >= 1 query outright, prune
the aggregate-dominated ones, then sample the parent in proportion to the number
of queries it is sole-best on (optionally discounted by 1/(1+children) so
under-explored members stay sampled -- Phase D3). This is what lets the search
escape the single hill run-5 stalled on at step 18.

The per-query signal is reciprocal rank (`mrr`) from MetricVector.per_query.
Members scored on different rotating-gate epochs simply contribute only the
queries they were scored on -- no cross-epoch rescoring needed.
"""
from ..artifact.program import SearchProgram
from ..reward.objectives import MetricVector
from ..reward.pareto import dominates

_INSTANCE_KEY = "mrr"  # per-query axis used for sole-best counting


class PoolMember:
    def __init__(self, program, metrics):
        self.program = program
        self.metrics = metrics
        self.children = 0         # times selected as a parent (Phase D3 discount)

    @property
    def sha(self):
        return self.program.sha


def _sole_best_counts(members):
    """For each member, the number of gate queries on which it is the UNIQUE
    maximum of the per-query instance score. Ties (incl. the common rr=0 on
    unreachable queries) award nobody. -> list[int] aligned with `members`."""
    counts = [0] * len(members)
    if len(members) < 1:
        return counts
    idxs = set()
    for m in members:
        idxs.update(getattr(m.metrics, "per_query", {}) or {})
    for idx in idxs:
        best_val, best_i, ties = None, -1, 0
        for i, m in enumerate(members):
            pq = (getattr(m.metrics, "per_query", {}) or {}).get(idx)
            if pq is None:
                continue
            v = pq.get(_INSTANCE_KEY, 0.0)
            if best_val is None or v > best_val:
                best_val, best_i, ties = v, i, 1
            elif v == best_val:
                ties += 1
        if best_i >= 0 and ties == 1 and best_val > 0.0:
            counts[best_i] += 1
    return counts


class CandidatePool:
    def __init__(self, cap=24, max_tokens=0.0):
        self.cap = int(cap)
        # Hard bloat wall (run10c fix): a program whose source-token size exceeds
        # max_tokens is never admitted to the pool -- so it can never become a
        # parent and propagate the bloat. 0 = off. Mirrors the gate's token wall;
        # the gate guards the headline best, this guards the breeding population.
        self.max_tokens = float(max_tokens or 0.0)
        self.members = []

    def __len__(self):
        return len(self.members)

    def shas(self):
        return {m.sha for m in self.members}

    # ------------------------------------------------------------- admission

    def consider(self, program, metrics):
        """Admit `program` if non-dominated OR sole-best on >= 1 query.
        -> (frontier_grew, admitted): `frontier_grew` is True only when it is a
        NEW non-dominated (aggregate Pareto) point -- that is the signal the
        loop's stop condition counts (A4: stale = steps adding no frontier
        member). A specialist admitted only on the instance-wise rule does NOT
        grow the frontier."""
        # Hard-reject crashes BEFORE the Pareto test. A crashed candidate aborts
        # early, so its cost axes are garbage-LOW (e.g. db_load drops because it
        # never finished the DB work) -- on the cost-minimizing axes of dominates()
        # that makes a mcq=0 crash "non-dominated", so it would slip onto the
        # frontier and (worse) reset the stale counter via frontier_grew. A broken
        # program is never a frontier member.
        if getattr(metrics, "crashed", False):
            return False, False
        # Hard bloat wall: over-cap programs never enter the breeding pool (so they
        # can't become parents) -- the run10c bloat-spiral fix. Enforced here, not
        # advisory: pool.consider previously ignored size entirely.
        if (self.max_tokens
                and getattr(metrics, "code_tokens", 0.0) > self.max_tokens):
            return False, False
        if program.sha in self.shas():
            return False, False
        dominated = any(dominates(m.metrics, metrics) for m in self.members)
        frontier = not dominated
        if not frontier:
            # would it be sole-best on >=1 query if added?
            hypo = self.members + [PoolMember(program, metrics)]
            sole = _sole_best_counts(hypo)[-1]
            if sole < 1:
                return False, False
        # admit: drop everything this candidate dominates, then append + cap.
        self.members = [m for m in self.members if not dominates(metrics, m.metrics)]
        self.members.append(PoolMember(program, metrics))
        self._evict()
        # `program` may have been evicted again if the frontier already filled
        # the cap; report frontier growth only if it survived.
        survived = program.sha in self.shas()
        return (frontier and survived), survived

    def _frontier(self, members=None):
        ms = self.members if members is None else members
        return [m for m in ms
                if not any(dominates(o.metrics, m.metrics) for o in ms if o is not m)]

    def _evict(self):
        if len(self.members) <= self.cap:
            return
        front = self._frontier()
        front_shas = {id(m) for m in front}
        rest = [m for m in self.members if id(m) not in front_shas]
        rest.sort(key=lambda m: m.metrics.primary, reverse=True)
        keep = front + rest[: max(0, self.cap - len(front))]
        if len(keep) > self.cap:  # frontier alone exceeds the cap: keep the best
            keep = sorted(front, key=lambda m: m.metrics.primary, reverse=True)[: self.cap]
        self.members = keep

    # ------------------------------------------------------------- selection

    def select_parent(self, rng, discount=True):
        """GEPA Alg. 2: sample a parent in proportion to the number of queries it
        is sole-best on (members winning 0 queries are not parents); prune
        aggregate-dominated winners; fall back to the highest-primary member when
        nobody is sole-best anywhere (e.g. an all-ties early pool)."""
        if not self.members:
            return None
        counts = _sole_best_counts(self.members)
        winners = [(m, c) for m, c in zip(self.members, counts) if c >= 1]
        if not winners:
            return max(self.members, key=lambda m: m.metrics.primary)
        kept = [(m, c) for m, c in winners
                if not any(dominates(o.metrics, m.metrics)
                           for o, _ in winners if o is not m)]
        if not kept:
            kept = winners
        weights = [c * (1.0 / (1 + m.children) if discount else 1.0) for m, c in kept]
        total = sum(weights)
        if total <= 0:
            chosen = max(self.members, key=lambda m: m.metrics.primary)
        else:
            r = rng.random() * total
            acc, chosen = 0.0, kept[-1][0]
            for (m, _), w in zip(kept, weights):
                acc += w
                if r <= acc:
                    chosen = m
                    break
        chosen.children += 1
        return chosen

    def select_mate(self, rng, exclude_sha):
        """Pick a SECOND, distinct member to COMBINE with the chosen parent
        (run10c combine mode -- KernelEvolve sibling-insight, NOT GA crossover).
        Prefer a complementary specialist: sample among the other members in
        proportion to the queries they win outright (so the mate brings skills the
        parent may lack); fall back to the highest-primary distinct member. Returns
        None when the pool has no other member. Does NOT bump `children` -- the
        mate is a reference, not the lineage parent."""
        others = [m for m in self.members if m.sha != exclude_sha]
        if not others:
            return None
        by_sha = {m.sha: c for m, c in zip(self.members, _sole_best_counts(self.members))}
        winners = [(m, by_sha.get(m.sha, 0)) for m in others if by_sha.get(m.sha, 0) >= 1]
        if not winners:
            return max(others, key=lambda m: m.metrics.primary)
        total = sum(c for _, c in winners)
        r = rng.random() * total
        acc, chosen = 0.0, winners[-1][0]
        for m, c in winners:
            acc += c
            if r <= acc:
                chosen = m
                break
        return chosen

    def best(self, key=None):
        """Highest member by `key(metrics)` (default: recall@20 primary)."""
        if not self.members:
            return None
        key = key or (lambda mv: mv.primary)
        return max(self.members, key=lambda m: key(m.metrics))

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        """Serializable snapshot for the Phase-1 campaign checkpoint. Stores each
        member's program SOURCE (the sha is derived) + family + full MetricVector
        (incl. per_query, which the score_cache's flat form drops) + children
        count, so a resumed run rebuilds the exact selection state.

        Artifact-agnostic: a FileSet (graph_search target) carries `to_dict`, so
        its overlay is stored under "artifact" with kind="file_set"; a
        SearchProgram (function target) stores "src"+"family" as before -- the
        function-campaign checkpoint is byte-identical."""
        members = []
        for m in self.members:
            rec = {"metrics": m.metrics.to_dict(), "children": m.children}
            if hasattr(m.program, "overlay"):  # FileSet
                rec["kind"] = "file_set"
                rec["artifact"] = m.program.to_dict()
            else:                               # SearchProgram
                rec["src"] = m.program.src
                rec["family"] = m.program.family
            members.append(rec)
        return {"cap": self.cap, "members": members}

    @classmethod
    def from_dict(cls, d: dict, validate=None) -> "CandidatePool":
        """Rebuild from `to_dict()`. `validate(program) -> bool` (optional) drops
        any member whose program no longer loads -- mirrors openEvolve's defensive
        reload (prune broken programs rather than abort the whole resume), but
        without inheriting its monolith. Member order/children are preserved.

        `validate` is called with the rebuilt program object (FileSet) on the
        graph_search path and with the source STRING on the function path, since
        the existing function-campaign validate (`_compiles`) takes a src string;
        the loop passes a validate that handles whichever it built."""
        pool = cls(cap=int(d.get("cap", 24)))
        for rec in d.get("members", []):
            if rec.get("kind") == "file_set":
                from ..artifact.file_set import FileSet
                prog = FileSet.from_dict(rec["artifact"])
                check_arg = prog
            else:
                prog = SearchProgram(rec["src"],
                                     family=rec.get("family", "unknown"))
                check_arg = rec["src"]
            if validate is not None and not validate(check_arg):
                continue
            member = PoolMember(prog, MetricVector.from_dict(rec["metrics"]))
            member.children = int(rec.get("children", 0))
            pool.members.append(member)
        return pool
