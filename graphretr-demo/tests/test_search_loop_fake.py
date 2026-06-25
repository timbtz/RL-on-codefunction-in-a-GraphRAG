"""Offline full-loop test for the graph_search target (plan Task 15): mutate ->
multi-file edit -> MCQ score -> Pareto/gate/checkpoint, with FakeSearchTarget +
StubAnswerer and a STUB mutator agent. No Neo4j, no API keys, no MLflow server.

This exercises the integration seams the real campaign uses (FileSet through the
pool/checkpoint, McqRewardAdapter through FastLoop's RewardModel call sites, the
NullSandbox/NullGraph, _reflect on MCQ rows) without any infra.
"""
import contextlib
import os
import shutil
from dataclasses import replace

import pytest

from graphretr_opt.campaign import SEARCH_DOMAIN_NOTE, Campaign
from graphretr_opt.config import load_config
from graphretr_opt.optimizer.edit_budget import EditBudget
from graphretr_opt.optimizer.fast_loop import FastLoop
from graphretr_opt.optimizer.mutator import Mutator

# A canned edit that matches the real service file exactly once.
_CANNED_EDIT = (
    "Diagnosis: widen neighbour fan-out to surface more documents.\n"
    "FILE: common/service/search/agentic_graph_traversal_search_service.py\n"
    "<<<<<<< SEARCH\n        neighbor_limit: int = 25,\n=======\n"
    "        neighbor_limit: int = 30,\n>>>>>>> REPLACE\n")


class StubAgent:
    """A mutator agent that returns a fixed multi-file edit (no LLM)."""
    call_counts = {}

    def digest(self, evidence):
        return evidence

    def complete(self, prompt, tier="editor"):
        return _CANNED_EDIT


class NoopTracker:
    """Implements only what FastLoop touches; no MLflow."""
    def log_vector(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_table(self, *a, **k): pass
    def log_dataset(self, *a, **k): pass
    def log_artifacts(self, *a, **k): pass

    @contextlib.contextmanager
    def step_span(self, *a, **k):
        yield None

    def record_step(self, *a, **k): pass


def test_fake_loop_runs_and_checkpoints():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset = os.path.join(repo_root, "graphsearch", "data", "dataset.json")
    if not os.path.exists(dataset):
        pytest.skip("offline MCQ fixture graphsearch/data/dataset.json missing "
                    "(run: python -m graphsearch.mcq_gen.build --offline)")

    cfg = load_config(target="graph_search", fake_target=True)
    camp = Campaign(cfg).boot_search()      # builds fake target + stub answerer + seed
    cfg = replace(camp.cfg, steps=3, checkpoint_every=1, stop_after_stale=0,
                  repair_budget=0)

    mutator = Mutator(StubAgent(), SEARCH_DOMAIN_NOTE, safe_builtins={})
    edit_budget = EditBudget(cfg.edit_schedule, cfg.max_edits, cfg.min_edits, 3)
    run_dir = os.path.join(cfg.runs_dir, "pytest_fake_loop")
    shutil.rmtree(run_dir, ignore_errors=True)
    try:
        loop = FastLoop(cfg, camp.search_graph, camp.search_sandbox,
                        camp.search_reward, mutator, edit_budget, NoopTracker(),
                        budget=None)
        result = loop.run(camp.search_substrate, camp.search_seed, 3,
                          "pytest_fake_loop")
        # the loop completed and produced an artifact + checkpoint
        assert result.best_program is not None
        assert result.best_metrics is not None
        assert "mcq_accuracy" in result.best_metrics.quality
        assert os.path.exists(os.path.join(run_dir, "checkpoint.json"))
        assert os.path.exists(os.path.join(run_dir, "best_search.py"))
        # the canned edit was applied at least once (step_000 candidate written)
        assert os.path.exists(os.path.join(run_dir, "step_000.py"))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
