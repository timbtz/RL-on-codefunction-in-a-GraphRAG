"""cli.py -- the three entrypoints: stage0-probe | optimize | final-test.

    python -m graphretr_opt.cli stage0   [--campaign-name stage0]
    python -m graphretr_opt.cli optimize [--steps N] [--campaign-name campaign]
    python -m graphretr_opt.cli final    --campaign-name campaign
    python -m graphretr_opt.cli ablate   [--strategies a,b,c] [--test-n N]

Config comes from configs/campaign.yaml + env vars (see config.py); only the
campaign name and step count are CLI flags.
"""
import argparse
import os

from .campaign import Campaign
from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="graphretr_opt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("stage0", help="seed vs one-shot headroom check")
    p0.add_argument("--campaign-name", default="stage0")

    po = sub.add_parser("optimize", help="run the fast-loop campaign (no test)")
    po.add_argument("--steps", type=int, default=None)
    po.add_argument("--campaign-name", default="campaign")
    po.add_argument("--strategy", default=None,
                    help="override the campaign.yaml strategy/seed family")
    po.add_argument("--resume", action="store_true",
                    help="resume from runs/<campaign>/checkpoint.json if present")
    po.add_argument("--checkpoint-every", type=int, default=None,
                    help="atomically snapshot the live pool every N steps (0=off)")

    ps = sub.add_parser("optimize-search",
                        help="evolve the REAL agentic search service (graph_search "
                             "target): edit real files, score via subprocess + MCQ")
    ps.add_argument("--steps", type=int, default=None)
    ps.add_argument("--campaign-name", default="search")
    ps.add_argument("--resume", action="store_true",
                    help="resume from runs/<campaign>/checkpoint.json if present")
    ps.add_argument("--checkpoint-every", type=int, default=None,
                    help="atomically snapshot the live pool every N steps (0=off)")
    ps.add_argument("--fake-target", action="store_true",
                    help="use FakeSearchTarget (offline: no Neo4j / API keys)")

    pf = sub.add_parser("final", help="score seed+best on the locked test split once")
    pf.add_argument("--campaign-name", required=True)
    pf.add_argument("--strategy", default=None,
                    help="must match the strategy the campaign was run with")
    pf.add_argument("--test-n", type=int, default=0,
                    help="score a deterministic N-query test subsample (0 = full split)")

    pa = sub.add_parser("ablate", help="seed-only embedder-vs-extractor attribution")
    pa.add_argument("--strategies", default="vector_only,hybrid_rrf,extract_first",
                    help="comma-separated seed families to score on the same gate")
    pa.add_argument("--test-n", type=int, default=0,
                    help="also score on a fixed N-query test subsample (0 = skip)")
    pa.add_argument("--campaign-name", default="ablate")

    pv = sub.add_parser("viz", help="render runs/<campaign>/lineage.jsonl to a "
                                    "self-contained lineage.html (read-only)")
    pv.add_argument("--campaign-name", required=True)
    pv.add_argument("--serve", action="store_true",
                    help="run a live Flask server (re-reads on each load) instead "
                         "of writing the static HTML")
    pv.add_argument("--port", type=int, default=8000)
    pv.add_argument("--out", default=None, help="static output path (default: "
                    "runs/<campaign>/lineage.html)")

    args = ap.parse_args(argv)
    if args.cmd == "viz":
        # read-only; no Campaign.boot() (no FalkorDB / embedder needed)
        from .config import load_config as _lc
        from .viz.lineage_viz import export_static, serve
        run_dir = os.path.join(_lc().runs_dir, args.campaign_name)
        if args.serve:
            serve(run_dir, port=args.port)
        else:
            print(f"[viz] wrote {export_static(run_dir, args.out)}")
        return
    overrides = ({"strategy": args.strategy}
                 if getattr(args, "strategy", None) else {})
    if getattr(args, "resume", False):
        overrides["resume"] = True
    if getattr(args, "checkpoint_every", None) is not None:
        overrides["checkpoint_every"] = args.checkpoint_every

    if args.cmd == "optimize-search":
        # graph_search path: boot_search() (no FalkorDB / stark_qa / torch).
        overrides["target"] = "graph_search"
        if getattr(args, "fake_target", False):
            overrides["fake_target"] = True
        camp = Campaign(load_config(**overrides)).boot_search()
        camp.optimize_search(steps=args.steps, campaign=args.campaign_name)
        return

    campaign = Campaign(load_config(**overrides)).boot()

    if args.cmd == "stage0":
        campaign.stage0(args.campaign_name)
    elif args.cmd == "optimize":
        campaign.optimize(steps=args.steps, campaign=args.campaign_name)
    elif args.cmd == "final":
        campaign.final_test(args.campaign_name, test_n=args.test_n)
    elif args.cmd == "ablate":
        campaign.ablate(tuple(s.strip() for s in args.strategies.split(",") if s.strip()),
                        test_n=args.test_n, campaign=args.campaign_name)


if __name__ == "__main__":
    main()
