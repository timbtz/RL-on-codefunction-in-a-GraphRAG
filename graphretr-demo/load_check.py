#!/usr/bin/env python
"""
load_check.py  -- Stage 1a verification that STaRK 'prime' loads in Python.

Loads the prime semi-structured knowledge base (SKB) and the prime QA dataset
from HuggingFace (processed download), then prints the structural facts the
next stage needs: node/edge counts, node types, schema triples, a sample node
document, a sample query + its gold answer_ids, and the train/val/test sizes.

API note (verified against installed stark_qa==1.1.0, not the briefing's
pseudocode): qa[i] returns a 4-tuple (query, q_id, answer_ids, meta_info),
NOT an object with .query / .answer_ids attributes.
"""
import time
from stark_qa import load_qa, load_skb


def main():
    t0 = time.time()
    print("Loading SKB 'prime' (download_processed=True; first run downloads from HF ~5 min)...")
    skb = load_skb("prime", download_processed=True)
    print(f"  SKB loaded in {time.time() - t0:.1f}s")

    print("Loading QA 'prime'...")
    qa = load_qa("prime")

    n_nodes = skb.num_nodes()
    n_edges = skb.num_edges()
    print("\n========== STaRK prime SKB ==========")
    print(f"node count : {n_nodes}")
    print(f"edge count : {n_edges}  (edge_index shape: {tuple(skb.edge_index.shape)})")

    # skb.node_types is a per-node tensor of type-ids; the human-readable
    # mapping is node_type_dict (id -> name).
    print(f"node_types (id -> name): {skb.node_type_dict}")
    print(f"edge_types (id -> name): {skb.edge_type_dict}")
    print(f"# distinct node labels: {len(skb.node_type_dict)} | # distinct edge types: {len(skb.edge_type_dict)}")

    # Per-label node counts (drives ETL batching + vector-index-per-label).
    print("\nnodes per label:")
    for tid, tname in skb.node_type_dict.items():
        cnt = int((skb.node_types == tid).sum())
        print(f"  {tname:24s} {cnt}")

    print("\n========== schema triples: skb.get_tuples() ==========")
    tuples = skb.get_tuples()
    print(f"# distinct (src_type, rel, dst_type) schema triples: {len(tuples)}")
    for t in tuples:
        print(f"  {t[0]} -[{t[1]}]-> {t[2]}")

    print("\n========== sample node: skb.get_doc_info(0) ==========")
    sample_id = 0
    print(f"node id {sample_id}  type={skb.get_node_type_by_id(sample_id)}")
    print(f"node_info[{sample_id}] keys: {list(skb.node_info[sample_id].keys())}")
    doc = skb.get_doc_info(sample_id)
    print("get_doc_info ->")
    print(doc[:800] + ("..." if len(doc) > 800 else ""))

    print(f"\ncandidate_ids: {len(skb.candidate_ids)} (answerable nodes; min={min(skb.candidate_ids)} max={max(skb.candidate_ids)})")

    print("\n========== QA dataset ==========")
    print(f"total queries: {len(qa)}")
    split = qa.get_idx_split()
    for k, v in split.items():
        print(f"  split '{k}': {len(v)} queries")

    print("\n========== sample query (first train query) ==========")
    train_pos = int(split["train"][0])
    query, q_id, answer_ids, meta_info = qa[train_pos]
    print(f"split-position {train_pos} -> q_id {q_id}")
    print(f"query: {query}")
    print(f"answer_ids (gold, {len(answer_ids)}): {answer_ids}")
    # Confirm gold answers are valid candidate node-ids.
    bad = [a for a in answer_ids if a not in set(skb.candidate_ids)]
    print(f"gold answers that are NOT candidate_ids: {bad if bad else 'none (all valid)'}")
    for a in answer_ids[:1]:
        print(f"  gold node {a} type={skb.get_node_type_by_id(a)} name={skb.node_info[a].get('name')}")

    print(f"\nload_check OK in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
