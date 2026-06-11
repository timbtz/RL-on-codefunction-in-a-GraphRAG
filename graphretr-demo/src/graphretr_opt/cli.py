"""cli.py -- the three entrypoints: stage0-probe | optimize | final-test.

    python -m graphretr_opt.cli stage0   [--campaign-name stage0]
    python -m graphretr_opt.cli optimize [--steps N] [--campaign-name campaign]
    python -m graphretr_opt.cli final    --campaign-name campaign

Config comes from configs/campaign.yaml + env vars (see config.py); only the
campaign name and step count are CLI flags.
"""
import argparse

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

    pf = sub.add_parser("final", help="score seed+best on the locked test split once")
    pf.add_argument("--campaign-name", required=True)

    args = ap.parse_args(argv)
    campaign = Campaign(load_config()).boot()

    if args.cmd == "stage0":
        campaign.stage0(args.campaign_name)
    elif args.cmd == "optimize":
        campaign.optimize(steps=args.steps, campaign=args.campaign_name)
    elif args.cmd == "final":
        campaign.final_test(args.campaign_name)


if __name__ == "__main__":
    main()
