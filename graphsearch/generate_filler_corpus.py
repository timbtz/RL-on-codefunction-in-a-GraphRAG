"""Grow the graph ~10x with DISJOINT filler projects so brute-force "load the
whole graph" stops working and the optimizer must retrieve selectively.

Each filler project is a full cross-source chain (tickets -> standup -> PRs ->
doc + chat) using vocabulary DISJOINT from the curated 5 projects (gen-* people,
components, GEN-* tickets, gen-* repos/docs/channels). The curated corpus and the
96-question exam are left untouched; filler people/components are declared in
people.json / components.json (the ingestion now merges those with its defaults).

Re-runnable: strips any existing gen-/GEN- records first, then regenerates.
"""
import json
import os

HERE = os.path.dirname(__file__)
CORPUS = os.path.join(HERE, "data", "corpus")
DOCS = os.path.join(CORPUS, "docs")

N_PROJECTS = 65          # ~10 owners each -> ~650 filler owner nodes (~10x the 68 curated)
N_PEOPLE = 50
N_COMPONENTS = 25
DATES = [f"2026-0{m}-{d:02d}" for m in (3, 4, 5, 6) for d in (3, 9, 14, 21, 27)]

people = [{"handle": f"gen-e{n:03d}", "name": f"Engineer{n:03d}"} for n in range(N_PEOPLE)]
components = [{"slug": f"gen-c{n:03d}", "name": f"gen-c{n:03d}"} for n in range(N_COMPONENTS)]


def pick(seq, i):
    return seq[i % len(seq)]


def gen():
    tickets, meetings, prs, docs, chat = [], [], [], [], []
    chat_seq = {}
    for p in range(N_PROJECTS):
        comp = pick(components, p)["slug"]
        repo = f"gen-r{p % 12:02d}"
        owner = pick(people, p)
        rev1 = pick(people, p + 7)
        rev2 = pick(people, p + 13)
        rep = pick(people, p + 3)
        doc_stem = f"gen-doc-{p:03d}"
        base_key = 1000 + p * 3
        t_keys = [f"GEN-{base_key + k}" for k in range(3)]  # 3 tickets, internal blocker chain

        # doc (Confluence)
        docs.append((doc_stem, owner["handle"], comp,
                     f"Design note for {comp} workstream {p}. Owner {owner['name']}. "
                     f"Covers the {comp} rollout, dependencies, and rollback plan."))
        # tickets (Jira) with an internal BLOCKS chain
        for k, key in enumerate(t_keys):
            tickets.append({
                "key": key,
                "title": f"{comp} task {k} for workstream {p}",
                "description": f"Work item {k} on the {comp} component (workstream {p}).",
                "status": pick(["Open", "In Progress", "Done"], p + k),
                "assignee": owner["handle"], "reporter": rep["handle"],
                "components": [comp],
                "references": [doc_stem],
                "blocks": [t_keys[k - 1]] if k > 0 else [],
            })
        # Teams standup
        meetings.append({
            "project": comp, "date": pick(DATES, p),
            "title": f"{comp} standup {p}",
            "attendees": [owner["handle"], rev1["handle"], rep["handle"]],
            "discusses": [t_keys[0]], "components": [comp], "references": [doc_stem],
            "transcript": [
                {"speaker": owner["handle"], "text": f"working through {t_keys[0]} on {comp}."},
                {"speaker": rev1["handle"], "text": "looks fine, ship it after review."},
            ],
        })
        # GitHub PRs (2) fixing the tickets
        for k in range(2):
            prs.append({
                "repo": repo, "number": 100 + p * 2 + k,
                "title": f"Implement {comp} task {k} (workstream {p})",
                "description": f"Resolves {t_keys[k]} on {comp}.",
                "state": pick(["merged", "open"], p + k),
                "author": owner["handle"], "reviewers": [rev1["handle"], rev2["handle"]],
                "fixes": [t_keys[k]], "components": [comp], "references": [doc_stem],
            })
        # chat (own channel)
        ch = f"gen{p % 8}"
        s = chat_seq.get(ch, 0)
        for k in range(3):
            chat.append({
                "seq": s + k + 1, "channel": ch, "author": pick(people, p + k)["handle"],
                "ts": f"{pick(DATES, p)}T0{k}:00:00Z",
                "text": f"@{owner['handle']} pushed [[{doc_stem}]] for {t_keys[0]} on {comp}.",
            })
        chat_seq[ch] = s + 3
    return tickets, meetings, prs, docs, chat


def strip_filler(arr, key):
    return [r for r in arr if not str(r.get(key, "")).startswith(("gen-", "GEN-"))]


def load(name):
    p = os.path.join(CORPUS, name)
    return json.load(open(p)) if os.path.exists(p) else []


def dump(name, obj):
    json.dump(obj, open(os.path.join(CORPUS, name), "w"), indent=1)


def main():
    tickets, meetings, prs, docs, chat = gen()

    # vocab files (merged with hardcoded defaults by ingest.ts)
    dump("people.json", people)
    dump("components.json", components)

    # tickets / meetings / github / chat: strip old filler, append fresh
    cur_t = strip_filler(load("tickets.json"), "key")
    dump("tickets.json", cur_t + tickets)

    cur_m = [r for r in load("meetings.json") if not str(r.get("project", "")).startswith("gen-")]
    dump("meetings.json", cur_m + meetings)

    cur_g = [r for r in load("github.json") if not str(r.get("repo", "")).startswith("gen-")]
    dump("github.json", cur_g + prs)

    cur_c = [r for r in load("chat.json") if not str(r.get("channel", "")).startswith("gen")]
    dump("chat.json", cur_c + chat)

    # filler docs (strip old gen-doc-*.md, write fresh)
    for f in os.listdir(DOCS):
        if f.startswith("gen-doc-"):
            os.remove(os.path.join(DOCS, f))
    for stem, author, comp, body in docs:
        with open(os.path.join(DOCS, stem + ".md"), "w") as fh:
            fh.write(f"---\nid: doc:{stem}\ntitle: {stem}\nauthor: {author}\n"
                     f"components: [{comp}]\n---\n# {stem}\n{body}\n")

    print(f"filler: {N_PROJECTS} projects, {len(people)} people, {len(components)} components")
    print(f"  +{len(tickets)} tickets, +{len(meetings)} meetings, +{len(prs)} PRs, "
          f"+{len(docs)} docs, +{len(chat)} chat msgs")
    print("curated 5 projects + 96-question exam left untouched.")


if __name__ == "__main__":
    main()
