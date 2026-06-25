"""mcq_format -- the shared letter-choice convention for the MCQ answerer.

Used by the generator's closed-book baseline pass. The optimizer's McqReward
deliberately re-implements the SAME tiny convention rather than importing this
module, so the hot loop never depends on graphsearch being importable -- but the
format (letters A.., last option = "cannot be determined", strict single-letter
parse) must stay in sync, so it is documented here as the source of truth.
"""
import re
import string

# The explicit escape hatch that guards the 1/N guessing floor: when the context
# does not support any option, the answerer is told to pick THIS one.
CANNOT_DETERMINE = "Kann aus dem Kontext nicht bestimmt werden."

ANSWER_SYSTEM = (
    "You answer a multiple-choice question using ONLY the provided context. "
    "Choose the single best option. If the context does not contain enough "
    "information to decide, choose the option that says it cannot be determined. "
    "Reply with EXACTLY one capital letter (A, B, C, ...) and nothing else."
)


def letters(n):
    return list(string.ascii_uppercase[:n])


def render_choices(choices) -> str:
    return "\n".join(f"{l}) {c}" for l, c in zip(letters(len(choices)), choices))


def render_user(question, choices, context=None) -> str:
    ctx = context if context is not None else "(no context provided)"
    return (f"Context:\n{ctx}\n\nQuestion:\n{question}\n\n"
            f"Options:\n{render_choices(choices)}\n\n"
            "Answer with one letter:")


def parse_letter(text, n_choices) -> int | None:
    """Strict parse: the first standalone A.. letter within range -> 0-based
    index, else None (caller falls back to the cannot-determine option)."""
    if not text:
        return None
    m = re.search(r"\b([A-Za-z])\b", text)   # standalone letter only
    if not m:
        return None
    idx = string.ascii_uppercase.index(m.group(1).upper())
    return idx if 0 <= idx < n_choices else None
