"""test_stark_parity.py -- the carve-out's crux gate (plan step 7 / risk §8).

History: this gate originally compared the subprocess STaRK worker against the
in-process `Sandbox.run` and confirmed identical ranking (the gate that cleared
deletion of `sandbox.py`, 2026-06-25). With the in-process path now removed, it
is a REGRESSION guard: the subprocess STaRK path -- driven through the real
`StarkRewardAdapter -> StarkSubprocessTarget -> _worker_stark` stack used in
production -- must still reproduce the recorded golden ranking
(tests/golden/stark_parity_v7.json) for the `reasoning_first_v7` seed on a FIXED
gate query. Ranking equality is the true invariant (every gate metric is a
function of the id ranking); raw scores drift ~1e-5..1e-4 because
embedder=openai_small makes a live, non-bit-reproducible embeddings call.

REQUIRES live infra: FalkorDB on :6380 (the indexed STaRK-prime graph) AND a
WORKING embeddings key (embedder=openai_small routes query embeds through
OpenRouter). It is therefore OPT-IN: skipped unless STARK_PARITY=1 is set, and
all live work happens INSIDE the test (never at import/collection time) so the
offline suite collects and stays green regardless.

Run it (with infra up + a funded key):
    STARK_PARITY=1 PYTHONPATH=..:src .venv/bin/python -m pytest \
        tests/test_stark_parity.py -q
"""
import json
import os
import socket

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # graphretr-demo
REPO_ROOT = os.path.dirname(ROOT)
# The candidate is now a FileSet over starksearch/src; this is the editable
# service file ported from the reasoning_first_v7 seed (parity reference).
SEED_PATH = os.path.join(
    REPO_ROOT, "starksearch/src/stark_search/stark_graph_search_service.py")
GOLDEN = os.path.join(ROOT, "tests", "golden", "stark_parity_v7.json")

pytestmark = pytest.mark.skipif(
    os.environ.get("STARK_PARITY") != "1",
    reason="STaRK subprocess<->golden parity needs live FalkorDB + a funded "
           "embeddings key; set STARK_PARITY=1 to run.")


def _falkor_up(host="127.0.0.1", port=6380):
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def _fixed_query():
    """A single, recorded gate query id -> its question text. Uses the substrate's
    deterministic gate so the same query is scored both ways."""
    from starksearch.qa import Substrate
    sub = Substrate(meta_holdout_size=0)
    gate = sub.gate_idxs(os.path.join(ROOT, "runs", "_parity"), 1, 42)
    q, q_id, answer_ids = sub.example(gate[0])
    return q, q_id, list(answer_ids)


def _production_path(src, idx):
    """Score the candidate through the REAL production stack the optimizer uses on
    the STaRK target: StarkRewardAdapter -> StarkSubprocessTarget -> _worker_stark.
    Returns the per-query ranked id list (top of the pred), from the reflection row."""
    from graphretr_opt.config import load_config
    from graphretr_opt.env.targets.stark_subprocess_target import StarkSubprocessTarget
    from starksearch.qa import Substrate
    from starksearch.reward import StarkRewardAdapter

    cfg = load_config()
    sub = Substrate(meta_holdout_size=0)
    target = StarkSubprocessTarget(
        falkor_cfg={"host": cfg.falkor_host, "port": cfg.falkor_port,
                    "graph_name": cfg.graph_name},
        opt_src_dir=os.path.join(ROOT, "src"), repo_root=REPO_ROOT,
        cfg_overrides={"root": cfg.root}, query_concurrency=1)
    reward = StarkRewardAdapter(sub, target, cfg.crash_frac_limit,
                                default_timeout_s=cfg.probe_timeout_s)
    mv, rows = reward.score(None, [idx], src=src, return_rows=True,
                            per_query_timeout_s=cfg.probe_timeout_s)
    row = rows[0]
    assert row.get("error") is None, f"per-query error: {row['error']}"
    # full pred ranking, rebuilt from the row's retrieved+gold (row keeps top-20
    # 'retrieved' which is enough to compare against the golden's top-20 ranking).
    return [i for i, _ in row["retrieved"]], mv


def test_stark_subprocess_reproduces_golden():
    if not _falkor_up():
        pytest.skip("FalkorDB not reachable on :6380")
    from starksearch.qa import Substrate
    from graphretr_opt.config import load_config
    from graphretr_opt.artifact.file_set import FileSet
    # The candidate is the whole editable service tree (FileSet over
    # starksearch/src), exactly what the optimizer evolves -- not a src string.
    cfg = load_config()
    src = FileSet.from_base(cfg.stark_src_abs, cfg.stark_editable_files,
                            family="stark_search")
    _query, q_id, _answer_ids = _fixed_query()
    gate = Substrate(meta_holdout_size=0).gate_idxs(
        os.path.join(ROOT, "runs", "_parity"), 1, 42)

    sub_rank20, mv = _production_path(src, gate[0])

    golden = json.load(open(GOLDEN, encoding="utf-8"))
    assert golden["q_id"] == q_id, "fixed gate query drifted from the golden"
    gpred = {int(k): float(v) for k, v in golden["pred"].items()}
    rank = lambda p: [i for i, _ in sorted(p.items(), key=lambda t: (-t[1], t[0]))]
    gold_rank20 = rank(gpred)[:20]

    # The true invariant: the production subprocess path reproduces the golden
    # top-20 RANKING exactly (every gate metric is a function of this ranking).
    assert sub_rank20 == gold_rank20, (
        "subprocess ranking regressed from golden:\n"
        f"  golden={gold_rank20}\n  subproc={sub_rank20}")
    assert mv.crashed_frac == 0.0, "production scoring crashed on the fixed query"
