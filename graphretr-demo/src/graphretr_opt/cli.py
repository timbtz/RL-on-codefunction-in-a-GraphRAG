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
    po.add_argument("--seed-from-run", default=None,
                    help="seed this run from runs/<NAME>/best_search.py (a prior "
                         "run's champion) -- archipelago seed-chaining")
    po.add_argument("--seed-champion", default=None,
                    help="seed from an explicit champion file path (overrides "
                         "--seed-from-run)")
    po.add_argument("--merge", action="store_true",
                    help="warm-start the pool with the --merge-seed champions and "
                         "force COMBINE mode from step 0 (archipelago merge)")
    po.add_argument("--merge-seed", nargs="+", default=None,
                    help="champion paths OR bare run names (resolved to "
                         "runs/<NAME>/best_search.py) to warm-start for --merge")

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

    parc = sub.add_parser("archipelago",
                          help="DAG meta-orchestrator: chain + tournament-merge "
                               "many optimize runs from a YAML spec")
    parc.add_argument("--spec", required=True, help="path to the archipelago YAML spec")
    parc.add_argument("--dry-run", action="store_true",
                      help="plan the branches + merge bracket and write the DAG; "
                           "launch no optimizer runs")
    parc.add_argument("--resume", action="store_true",
                      help="reuse the ledger + skip runs that already have "
                           "select_holdout.json")

    args = ap.parse_args(argv)
    if args.cmd == "archipelago":
        # orchestrator only: it shells out to `optimize` subprocesses, so it needs
        # no FalkorDB / embedder / Campaign.boot() of its own.
        from .archipelago import run_campaign
        run_campaign(args.spec, dry_run=args.dry_run, resume=args.resume)
        return
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

    # Archipelago seed/merge resolution (optimize subcommand only): a bare run
    # name resolves to runs/<NAME>/best_search.py; an explicit path is used as-is.
    if (getattr(args, "seed_from_run", None) or getattr(args, "seed_champion", None)
            or getattr(args, "merge", False)):
        runs_dir = load_config().runs_dir

        def _champ(name_or_path):
            if os.sep in name_or_path or name_or_path.endswith(".py"):
                p = name_or_path
            else:
                p = os.path.join(runs_dir, name_or_path, "best_search.py")
            if not os.path.exists(p):
                raise SystemExit(f"champion not found: {p}")
            return p

        if getattr(args, "seed_champion", None):
            overrides["seed_champion_path"] = _champ(args.seed_champion)
        elif getattr(args, "seed_from_run", None):
            overrides["seed_champion_path"] = _champ(args.seed_from_run)
        if getattr(args, "merge", False):
            seeds = getattr(args, "merge_seed", None) or []
            overrides["merge"] = True
            overrides["merge_seed_paths"] = tuple(_champ(s) for s in seeds)

    if args.cmd == "optimize-search":
        # graph_search path: cfg.target drives boot() -> boot_search() (no
        # FalkorDB / stark_qa / torch on this path).
        overrides["target"] = "graph_search"
        if getattr(args, "fake_target", False):
            overrides["fake_target"] = True
        camp = Campaign(load_config(**overrides)).boot()
        camp.optimize_search(steps=args.steps, campaign=args.campaign_name)
        return

    # function (STaRK) path: cfg.target defaults to "function" -> _boot_function().
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
