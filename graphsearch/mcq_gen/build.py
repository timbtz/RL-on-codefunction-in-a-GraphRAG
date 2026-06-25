"""build.py -- produce the gold MCQ set graphsearch/data/dataset.json.

Pipeline (one time, offline; NOT the optimizer loop):
  1. read the free-text gold Q&A (question, answer, source);
  2. synthesize a multiple-choice item per question -- 3 plausible distractors +
     the explicit "cannot be determined from context" option -- with the LLM
     (--provider/--model) or deterministically (--offline, no LLM / no key);
  3. run the closed-book baseline pass ONCE (answerer only, NO graph/context):
     stamp closedbook_correct per item; optionally DROP items the model already
     answers closed-book (--drop-closedbook), which measure parametric knowledge,
     not retrieval (Risk R6);
  4. write JSON (diffable, hand-editable -- NOT Parquet):
     [{id, question, choices[], answer_idx, source, closedbook_correct, lang}].

Eval hygiene: choices / answer_idx live ONLY in this file; they never reach the
search service.
"""
import argparse
import json
import os
import random

from .mcq_format import (ANSWER_SYSTEM, CANNOT_DETERMINE, parse_letter,
                         render_user)

N_DISTRACTORS = 3  # -> 4 real options + the cannot-determine escape = N=5


# --------------------------------------------------------------------------- #
# Distractor synthesis                                                         #
# --------------------------------------------------------------------------- #

def _offline_distractors(idx, items, rng):
    """Deterministic distractors: other items' answers (truncated). No LLM, so
    closedbook_correct is left False -- this path is for fixtures/plumbing, not a
    trustworthy difficulty/leakage estimate (use the LLM path for real runs)."""
    pool = [it["answer"] for j, it in enumerate(items) if j != idx]
    rng.shuffle(pool)
    picked = []
    for a in pool:
        a = a.strip()
        if a and a not in picked:
            picked.append(a if len(a) <= 160 else a[:157] + "…")
        if len(picked) >= N_DISTRACTORS:
            break
    while len(picked) < N_DISTRACTORS:        # tiny corpora: pad with placeholders
        picked.append(f"Keine der genannten Optionen ({len(picked)})")
    return picked


def _llm_distractors(chat, item, rng):
    """Ask the model for graph-plausible distractors. Falls back to offline-style
    if the reply is unusable."""
    from .mcq_format import letters  # local: only needed on the LLM path
    sys_p = (
        "Generate exactly 3 plausible but INCORRECT short answer options for the "
        "given question, given the correct answer. They must be on-topic and "
        "tempting (same domain/entities) but wrong. Return ONLY a JSON list of 3 "
        "strings, no commentary."
    )
    user_p = json.dumps({"question": item["question"], "correct": item["answer"]},
                        ensure_ascii=False)
    try:
        raw = chat.invoke([{"role": "system", "content": sys_p},
                           {"role": "user", "content": user_p}]).content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").split("\n", 1)[-1]
        data = json.loads(raw)
        out = [str(x).strip() for x in data if str(x).strip()][:N_DISTRACTORS]
        if len(out) == N_DISTRACTORS:
            return out
    except Exception as e:  # noqa: BLE001 -- any failure -> deterministic fallback
        print(f"[mcq_gen] distractor LLM failed ({e}); using offline distractors")
    return None


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #

def _assemble_item(idx, item, distractors, rng):
    """Place the correct answer + distractors in a deterministic shuffled order,
    append CANNOT_DETERMINE as the last option, and record answer_idx."""
    options = [item["answer"].strip()] + list(distractors)
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    answer_idx = order.index(0)
    shuffled.append(CANNOT_DETERMINE)          # always the last option
    return {
        "id": f"q{idx:03d}",
        "question": item["question"],
        "choices": shuffled,
        "answer_idx": answer_idx,
        "source": item.get("source", ""),
        "lang": "de",
    }


def _closedbook_correct(chat, mcq):
    """Ask the answerer the MCQ with NO context; True iff it picks answer_idx."""
    msg = chat.invoke([
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": render_user(mcq["question"], mcq["choices"],
                                                context=None)},
    ]).content
    chosen = parse_letter(msg, len(mcq["choices"]))
    if chosen is None:
        chosen = len(mcq["choices"]) - 1       # cannot-determine fallback
    return chosen == mcq["answer_idx"]


def build(in_path, out_path, offline=False, provider="openai",
          model="gpt-4o-mini", seed=0, drop_closedbook=False):
    items = json.load(open(in_path, encoding="utf-8"))
    chat = None
    if not offline:
        chat = _build_chat(provider, model)

    dataset = []
    for idx, item in enumerate(items):
        rng = random.Random(seed * 1009 + idx)
        distractors = None
        if chat is not None:
            distractors = _llm_distractors(chat, item, rng)
        if distractors is None:
            distractors = _offline_distractors(idx, items, rng)
        mcq = _assemble_item(idx, item, distractors, rng)
        if chat is not None:
            mcq["closedbook_correct"] = bool(_closedbook_correct(chat, mcq))
        else:
            mcq["closedbook_correct"] = False
        dataset.append(mcq)

    if drop_closedbook:
        before = len(dataset)
        dataset = [m for m in dataset if not m["closedbook_correct"]]
        print(f"[mcq_gen] dropped {before - len(dataset)} closed-book-correct "
              f"(parametric-leakage) items")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    n_cb = sum(1 for m in dataset if m["closedbook_correct"])
    print(f"[mcq_gen] wrote {len(dataset)} items -> {out_path} "
          f"({n_cb} closed-book-correct){' [OFFLINE fixture]' if offline else ''}")
    return dataset


def _build_chat(provider, model):
    provider = provider.lower()
    if provider in ("openai", "azure_openai"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    raise ValueError(f"unknown provider {provider!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="graphsearch.mcq_gen.build")
    ap.add_argument("--in", dest="in_path", default="graphsearch/data/gold_qa.json")
    ap.add_argument("--out", dest="out_path", default="graphsearch/data/dataset.json")
    ap.add_argument("--offline", action="store_true",
                    help="deterministic, no LLM (fixture/plumbing only)")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-closedbook", action="store_true",
                    help="drop items the model answers correctly closed-book")
    args = ap.parse_args(argv)
    build(args.in_path, args.out_path, offline=args.offline,
          provider=args.provider, model=args.model, seed=args.seed,
          drop_closedbook=args.drop_closedbook)


if __name__ == "__main__":
    main()
