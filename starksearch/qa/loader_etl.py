"""loader_etl.py -- offline, one-time ETL of STaRK 'prime' into the graph DB.

The working, idempotent implementation lives at the project root as
`etl_prime.py` (it embeds nodes once to emb_cache.npy, loads nodes + per-label
vector indexes + typed edges, and self-heals a partial load). It is invoked
directly, not imported, because it is a standalone offline script:

    .venv/bin/python etl_prime.py        # safe to re-run; skips if graph full

This module is the curated-layout pointer to it; `main()` execs that script so
`python -m starksearch.qa.loader_etl` also works.
"""
import os
import runpy

# This file lives at <repo-root>/starksearch/qa/loader_etl.py; etl_prime.py lives
# in the optimizer checkout at <repo-root>/graphretr-demo/etl_prime.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ETL_SCRIPT = os.path.join(REPO_ROOT, "graphretr-demo", "etl_prime.py")


def main():
    runpy.run_path(ETL_SCRIPT, run_name="__main__")


if __name__ == "__main__":
    main()
