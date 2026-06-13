"""reembed_openai.py -- one-time switch of node embeddings to OpenAI
text-embedding-3-small (1536-d), in place, on the live FalkorDB graph.

Idempotent and resume-safe:
  1. drop the 10 per-label 384-d vector indexes (skipped if already gone)
  2. walk nodes in id order, embed texts via the metered OpenAIBudget,
     SET n.embedding -- progress persisted to runs/reembed_progress.json,
     so a crash/Ctrl-C resumes where it left off
  3. recreate the vector indexes at dimension 1536
  4. ensure the full-text index on (Entity, text) exists
After this, set `embedder: openai_small` in configs/campaign.yaml.
Rollback: the original MiniLM vectors are still in emb_cache.npy; re-running
etl_prime's embedding write-back restores them.

    .venv/bin/python reembed_openai.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from falkordb import FalkorDB

from graphretr_opt.config import load_config
from graphretr_opt.env.openai_client import OpenAIBudget, EMBED_DIM

BATCH = 256
PROGRESS = os.path.join(ROOT, "runs", "reembed_progress.json")


def main():
    cfg = load_config()
    g = FalkorDB(host=cfg.falkor_host, port=cfg.falkor_port).select_graph(cfg.graph_name)
    budget = OpenAIBudget(os.path.join(cfg.runs_dir, "openai_usage.json"),
                          ceiling_usd=cfg.openai_budget_usd)

    labels = sorted(r[0] for r in g.ro_query(
        "CALL db.labels() YIELD label RETURN label").result_set
        if r[0] != "Entity")
    n_total = g.ro_query("MATCH (n:Entity) RETURN count(n)").result_set[0][0]
    print(f"[reembed] {n_total} nodes, {len(labels)} labels, "
          f"${budget.spent_usd:.2f} already spent")

    # 1. drop old vector indexes (re-run safe: ignore "does not exist")
    for lb in labels:
        try:
            g.query(f"DROP VECTOR INDEX FOR (n:`{lb}`) ON (n.embedding)")
            print(f"[reembed] dropped vector index on {lb}")
        except Exception as e:
            print(f"[reembed] drop {lb}: {e}")

    # 2. re-embed in id order with persisted progress
    done = json.load(open(PROGRESS))["next_id"] if os.path.exists(PROGRESS) else 0
    if done:
        print(f"[reembed] resuming from id {done}")
    t0 = time.time()
    written = 0
    while True:
        rows = g.ro_query(
            "MATCH (n:Entity) WHERE n.id >= $lo RETURN n.id, n.text "
            "ORDER BY n.id LIMIT $lim", {"lo": done, "lim": BATCH}).result_set
        if not rows:
            break
        vecs = budget.embed([r[1] or " " for r in rows])
        payload = [{"id": int(r[0]), "emb": v} for r, v in zip(rows, vecs)]
        g.query("UNWIND $rows AS row MATCH (n:Entity {id: row.id}) "
                "SET n.embedding = vecf32(row.emb)", {"rows": payload})
        done = int(rows[-1][0]) + 1
        written += len(rows)
        json.dump({"next_id": done}, open(PROGRESS, "w"))
        if written % (BATCH * 20) == 0:
            rate = written / (time.time() - t0)
            print(f"[reembed] {written} nodes ({rate:.0f}/s, "
                  f"${budget.spent_usd:.2f} spent, "
                  f"eta {(n_total - written) / rate / 60:.0f} min)")
    print(f"[reembed] embedded+wrote {written} nodes in {(time.time()-t0)/60:.1f} min, "
          f"total spend ${budget.spent_usd:.2f}")

    # 3. recreate vector indexes at the new dimension
    for lb in labels:
        try:
            g.query(f"CREATE VECTOR INDEX FOR (n:`{lb}`) ON (n.embedding) "
                    f"OPTIONS {{dimension:{EMBED_DIM}, similarityFunction:'cosine'}}")
            print(f"[reembed] created {EMBED_DIM}-d vector index on {lb}")
        except Exception as e:
            print(f"[reembed] create {lb}: {e}")

    # 4. full-text index (idempotent)
    try:
        g.query("CALL db.idx.fulltext.createNodeIndex('Entity', 'text')")
        print("[reembed] created full-text index on (Entity, text)")
    except Exception as e:
        print(f"[reembed] fulltext index: {e}")

    print("[reembed] DONE. Now set `embedder: openai_small` in configs/campaign.yaml")


if __name__ == "__main__":
    main()
