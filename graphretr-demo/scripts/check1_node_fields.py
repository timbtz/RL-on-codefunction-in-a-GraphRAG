#!/usr/bin/env python
"""
CHECK 1 -- Is the ingestion space actually rich?

For a few sample nodes of EACH node type, dump:
  * skb.node_info[i]            -- the raw semi-structured attribute dict
  * skb.get_doc_info(i)         -- the text the ETL currently embeds (frozen choice)

Then quantify, per type, how many raw fields exist vs. how many of them
actually surface in the embedded document. If many useful fields are ignored,
there is room to search over ingestion (the thesis go/no-go premise).

Read-only. Loads STaRK 'prime' only -- no FalkorDB, no embedding.
"""
import re
import textwrap

from stark_qa import load_skb

SAMPLES_PER_TYPE = 2      # nodes to dump per node type
DOC_PREVIEW = 600         # chars of get_doc_info to show


def field_in_doc(value, doc_lower):
    """Heuristic: does this raw field's value appear in the embedded doc?"""
    if value is None:
        return False
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "[]", "{}"):
        return False
    # use the first reasonably-long token of the value as a probe
    probe = s.lower()[:40]
    return probe in doc_lower


def main():
    print("[check1] loading STaRK 'prime' SKB ...")
    skb = load_skb("prime", download_processed=True)
    n_nodes = skb.num_nodes()
    node_type_ids = skb.node_types.numpy()
    print(f"[check1] {n_nodes} nodes, {len(skb.node_type_dict)} node types\n")

    # Aggregate, per type, the union of field names seen and how often each is
    # reflected in the embedded doc, across the sampled nodes.
    summary = {}

    for tid, tname in skb.node_type_dict.items():
        ids = [int(i) for i in range(n_nodes) if node_type_ids[i] == tid]
        if not ids:
            continue
        sample = ids[:SAMPLES_PER_TYPE]
        print("=" * 88)
        print(f"NODE TYPE: {tname!r}   (type id {tid}, {len(ids)} nodes total)")
        print("=" * 88)

        seen_fields = set()
        reflected = set()

        for i in sample:
            info = skb.node_info[i]
            doc = skb.get_doc_info(i)
            doc_lower = doc.lower()

            print(f"\n--- node id {i} ---")
            print("RAW node_info keys + values:")
            if isinstance(info, dict):
                for k, v in info.items():
                    seen_fields.add(k)
                    vs = str(v)
                    if len(vs) > 200:
                        vs = vs[:200] + f" ... [{len(str(v))} chars]"
                    hit = "IN-DOC " if field_in_doc(v, doc_lower) else "MISSING"
                    if field_in_doc(v, doc_lower):
                        reflected.add(k)
                    print(f"   [{hit}] {k!r}: {vs}")
            else:
                print(f"   (node_info is not a dict: {type(info)} -> {str(info)[:200]})")

            print("\nget_doc_info (embedded text) preview:")
            preview = doc[:DOC_PREVIEW]
            print(textwrap.indent(preview + ("..." if len(doc) > DOC_PREVIEW else ""),
                                  "   "))
            print(f"   [doc length: {len(doc)} chars]")

        ignored = sorted(seen_fields - reflected)
        summary[tname] = (sorted(seen_fields), sorted(reflected), ignored)
        print(f"\n>>> type {tname!r}: {len(seen_fields)} raw fields, "
              f"{len(reflected)} reflected in doc, {len(ignored)} ignored")
        if ignored:
            print(f">>> IGNORED fields (room to search): {ignored}")
        print()

    print("\n" + "#" * 88)
    print("# CHECK 1 SUMMARY -- ingestion richness per node type")
    print("#" * 88)
    print(f"{'node type':28s} {'raw':>4s} {'in-doc':>7s} {'ignored':>8s}")
    total_ignored = 0
    for tname, (fields, refl, ignored) in summary.items():
        total_ignored += len(ignored)
        print(f"{tname:28s} {len(fields):4d} {len(refl):7d} {len(ignored):8d}")
    print("-" * 88)
    print(f"types with >=1 ignored field: "
          f"{sum(1 for _,(_,_,ig) in summary.items() if ig)}/{len(summary)}")
    print(f"total distinct ignored (type,field) pairs: {total_ignored}")
    print("\nVERDICT: ingestion space is RICH if multiple types ignore useful "
          "raw fields above.")


if __name__ == "__main__":
    main()
