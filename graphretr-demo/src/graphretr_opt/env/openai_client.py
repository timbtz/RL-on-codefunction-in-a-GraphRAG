"""OpenAIBudget -- the single metered gateway to OpenAI, shared by the
embedder and G.extract.

Every call goes through spend tracking: request/token counters and an
estimated-$ total, persisted to runs/openai_usage.json after each request.
When the estimate reaches the configured ceiling the next call raises
BudgetExceeded -- a hard wall, checked BEFORE the request, so a runaway
loop stops at the ceiling instead of discovering it on the invoice.
"""
import json
import os
import threading
import time

# $ per 1M tokens (input, output). Estimates for the ledger, not invoices.
# OpenRouter prices verified 2026-06-21; reconfirm on openrouter.ai/models when
# changing models (an unlisted model defaults to the pricey (1.0, 4.0) fallback
# below, which would trip the budget ceiling early).
PRICES = {
    # legacy OpenAI (kept for old ledgers / fallback to direct OpenAI)
    "text-embedding-3-small": (0.02, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    # OpenRouter -- retrieval LLM workhorse (Chinese OSS, reliable json_object)
    "deepseek/deepseek-chat": (0.20, 0.80),       # DeepSeek V3
    # OpenRouter -- embeddings. The SAME model the nodes were indexed with, just
    # served through OpenRouter => NO re-index needed (1536-d, $0.02/M).
    "openai/text-embedding-3-small": (0.02, 0.0),
}

# Embeddings are a MATCHED PAIR: query vectors (here) and node vectors (in the
# FalkorDB index) MUST use this exact model+dim. The graph is already indexed
# with text-embedding-3-small (1536-d); routing that same model via OpenRouter
# keeps the index valid -- no re-embed/re-index.
EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_DIM = 1536
# ~8191-token embedding input limit; chars/4 heuristic with margin.
EMBED_MAX_CHARS = 30_000
# API cap is 300k tokens per REQUEST (sum over inputs). Sequence-heavy node
# texts tokenize near 2 chars/token, so cap request payloads at 500k chars
# (~250k tokens worst-case) and split bigger batches into several requests.
EMBED_REQ_MAX_CHARS = 500_000


class BudgetExceeded(RuntimeError):
    pass


def _parse_json(content):
    """Tolerant json_object parse -> dict ({} if hopeless). Some OpenRouter
    providers wrap JSON in ```json ... ``` fences or add prose around it despite
    response_format; strip the fence and fall back to the first {...} span before
    giving up (a silent {} here would quietly degrade retrieval)."""
    if not content:
        return {}
    s = content.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s.strip("`")
        if s.endswith("```"):
            s = s[: s.rfind("```")]
        s = s.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except (json.JSONDecodeError, TypeError):
                pass
        return {}


def step_cost_delta(prev, cur, accepted, ceiling_usd=0.0):
    """Per-step OpenAI spend from two `OpenAIBudget.snapshot()`s, split by gate
    outcome -- the `$/accepted-edit` axis (GEPA/AlphaEvolve cost reporting).

    `accepted` routes this step's tokens to `tokens_accepted` (spend that bought
    an edit) vs `tokens_rejected` (spend burned on a rejected proposal). Returns
    a flat metrics dict for `tracker.log_metrics(..., step=step)`.
    """
    d_tok = ((cur["tokens_in"] + cur["tokens_out"])
             - (prev["tokens_in"] + prev["tokens_out"]))
    d_usd = round(cur["usd"] - prev["usd"], 6)
    out = {
        "tokens_accepted": d_tok if accepted else 0,
        "tokens_rejected": 0 if accepted else d_tok,
        "usd_step": d_usd,
        "usd_cumulative": cur["usd"],
    }
    if ceiling_usd:
        out["usd_vs_ceiling"] = cur["usd"] / float(ceiling_usd)
    return out


class OpenAIBudget:
    def __init__(self, ledger_path, ceiling_usd=5.0):
        self._ledger_path = ledger_path
        self._ceiling = float(ceiling_usd)
        self._lock = threading.Lock()
        self._client = None
        if os.path.exists(ledger_path):
            self._usage = json.load(open(ledger_path))
        else:
            self._usage = {"requests": 0, "tokens_in": 0, "tokens_out": 0,
                           "usd": 0.0, "by_model": {}}

    @property
    def spent_usd(self):
        return self._usage["usd"]

    def snapshot(self):
        """Immutable copy of the running ledger -- the boundary for per-step cost
        attribution. Take one before and one after a step; `step_cost_delta`
        turns the pair into per-step spend split by gate outcome."""
        with self._lock:
            return {"requests": self._usage["requests"],
                    "tokens_in": self._usage["tokens_in"],
                    "tokens_out": self._usage["tokens_out"],
                    "usd": self._usage["usd"],
                    "by_model": {m: dict(v) for m, v in self._usage["by_model"].items()}}

    def _api(self):
        if self._client is None:
            from openai import OpenAI
            # OpenRouter is OpenAI-API-compatible: just point base_url at it and
            # use the OpenRouter key. Falls back to direct OpenAI if no OpenRouter
            # base/key is set, so this stays a no-op for the old setup.
            base = os.environ.get("OPENROUTER_BASE_URL")
            if base:
                key = (os.environ.get("OPEN_ROUTER_API_KEY")
                       or os.environ.get("OPENROUTER_API_KEY"))
                if not key:
                    raise RuntimeError(
                        "OPENROUTER_BASE_URL set but no OPEN_ROUTER_API_KEY (.env)")
                self._client = OpenAI(base_url=base, api_key=key)
            else:
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY not set (put it in .env)")
                self._client = OpenAI()
        return self._client

    @staticmethod
    def _provider_prefs():
        """OpenRouter routing prefs for chat calls -- only applied when pointed at
        OpenRouter. We PIN a known-good provider (DeepInfra) rather than sort=price:
        the absolute-cheapest provider (StreamLake) returns json_object wrapped in
        markdown fences, which broke parsing. Pinning also keeps scoring reproducible
        across candidate programs. allow_fallbacks=True so a provider outage doesn't
        kill the run (the parser strips fences if a fallback misbehaves)."""
        if not os.environ.get("OPENROUTER_BASE_URL"):
            return None
        return {"order": ["deepinfra"], "allow_fallbacks": True,
                "require_parameters": True}

    def _check(self):
        if self._usage["usd"] >= self._ceiling:
            raise BudgetExceeded(
                f"OpenAI budget ceiling hit: ${self._usage['usd']:.2f} >= "
                f"${self._ceiling:.2f} (raise openai_budget_usd to continue)")

    def _record(self, model, tokens_in, tokens_out):
        p_in, p_out = PRICES.get(model, (1.0, 4.0))  # unknown model: assume pricey
        usd = tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out
        with self._lock:
            u = self._usage
            u["requests"] += 1
            u["tokens_in"] += tokens_in
            u["tokens_out"] += tokens_out
            u["usd"] = round(u["usd"] + usd, 6)
            m = u["by_model"].setdefault(model, {"requests": 0, "usd": 0.0})
            m["requests"] += 1
            m["usd"] = round(m["usd"] + usd, 6)
            os.makedirs(os.path.dirname(self._ledger_path), exist_ok=True)
            json.dump(u, open(self._ledger_path, "w"), indent=1)

    # ----------------------------------------------------------------- calls

    def embed(self, texts, model=EMBED_MODEL):
        """texts: list[str] -> list[list[float]] (unit-normalized by OpenAI)."""
        self._check()
        texts = [t[:EMBED_MAX_CHARS] if t else " " for t in texts]
        out, batch, chars = [], [], 0
        for t in texts:
            if batch and chars + len(t) > EMBED_REQ_MAX_CHARS:
                out.extend(self._embed_request(batch, model))
                batch, chars = [], 0
            batch.append(t)
            chars += len(t)
        if batch:
            out.extend(self._embed_request(batch, model))
        return out

    def _embed_request(self, texts, model, attempt=0):
        from openai import BadRequestError, RateLimitError
        try:
            resp = self._api().embeddings.create(model=model, input=texts)
        except RateLimitError:
            # TPM window (1M tok/min) clears in seconds; just wait it out.
            if attempt >= 8:
                raise
            time.sleep(min(2.0 * 2 ** attempt, 60.0))
            return self._embed_request(texts, model, attempt + 1)
        except BadRequestError:
            # Still over a token cap (char heuristics can't be exact): bisect
            # the batch; a single offending text gets truncated harder.
            if len(texts) == 1:
                if len(texts[0]) <= 1000:
                    raise
                return self._embed_request([texts[0][:len(texts[0]) // 2]], model)
            mid = len(texts) // 2
            return (self._embed_request(texts[:mid], model)
                    + self._embed_request(texts[mid:], model))
        self._record(model, resp.usage.prompt_tokens, 0)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def chat_json(self, system, user, model="gpt-4o-mini", max_tokens=400):
        """One JSON-mode chat call -> parsed dict ({} on unparseable output)."""
        self._check()
        kwargs = dict(
            model=model, max_tokens=max_tokens, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        prefs = self._provider_prefs()
        if prefs is not None:
            kwargs["extra_body"] = {"provider": prefs}
        resp = self._api().chat.completions.create(**kwargs)
        self._record(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return _parse_json(resp.choices[0].message.content)
