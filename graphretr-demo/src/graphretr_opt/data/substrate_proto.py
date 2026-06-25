"""substrate_proto.py -- the engine's `Substrate` contract (one named type).

The optimizer core (`fast_loop`, the per-target reward) consumes a "substrate":
the dataset adapter that hands out query indices, the (query, id, gold) example
tuple, the gate/test partitions, and the target's metric evaluator. Before the
carve-out this was duck-typed convention scattered across two implementations
(STaRK `starksearch.qa.Substrate`, company-KG `graphsearch.qa.QaSubstrate`).

This Protocol pins it so both services conform to ONE declared contract and the
`make_evaluator()` return-type split (a real STaRK Evaluator vs `None` for the
MCQ path) is explicit, not accidental. `@runtime_checkable` so a smoke test can
assert conformance with `isinstance`.
"""
from typing import Any, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class Evaluator(Protocol):
    """The minimal metric evaluator the STaRK reward calls. The MCQ path returns
    None instead (it scores via its own answerer), so reward code must treat the
    result as Optional."""

    def evaluate(self, pred: Any, answer_ids: Any, metrics: Sequence[str]) -> Any:
        ...


@runtime_checkable
class Substrate(Protocol):
    """Dataset adapter the optimizer evolves a search program against."""

    @property
    def train_idxs(self) -> Sequence[int]:
        """Positional indices the rollout/gate samplers draw from."""
        ...

    def example(self, idx: int) -> Tuple[str, Any, Sequence[int]]:
        """-> (query, q_id, answer_ids) for one positional index."""
        ...

    def gate_idxs(self, run_dir: str, size: int, seed: int) -> Sequence[int]:
        """The fixed gate subsample (persisted under run_dir on first use)."""
        ...

    def rotating_gate_idxs(self, size: int, seed: int, epoch: int) -> Sequence[int]:
        """A per-epoch rotating gate subsample (anti-overfit gate rotation)."""
        ...

    def get_test_idxs_I_KNOW_THIS_IS_FINAL(self) -> Sequence[int]:
        """The locked test split -- touched only by final_test/ablate."""
        ...

    def make_evaluator(self) -> Optional[Evaluator]:
        """The target's metric evaluator, or None when the target scores without
        one (e.g. the MCQ answerer path)."""
        ...
