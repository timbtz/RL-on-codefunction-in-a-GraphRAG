"""SingleCoder -- one reflective coding agent. Diagnoses misses and emits a
new program in a single combined call (SkillOpt style, reasoning model).

Backends:
  * 'cli': `claude -p --model <m>` subprocess -- zero config, works on a box
    with only the authenticated CLI.
  * 'sdk': the anthropic client when an ANTHROPIC_API_KEY is present (cleaner
    transcripts, token counts).
  * 'zai': z.ai's GLM (e.g. glm-5.2) via the openai SDK pointed at z.ai's
    OpenAI-compatible endpoint -- direct HTTPS, no CLI subprocess, lower
    latency per step. Needs ZAI_API_KEY (or Z.AI_KEY) in env.
"""
import os
import subprocess
import time
from collections import Counter

# z.ai (GLM) -- OpenAI-compatible. Point the openai SDK here (with the
# ZAI_API_KEY / Z.AI_KEY env var) to call GLM directly instead of the CLI.
# DEFAULT is the CODING endpoint (/api/coding/paas/v4): the optimizer is a code
# agent, and z.ai "Personal Coding Plan" keys (the kind Claude Code itself runs
# on) are authorized here. The general /api/paas/v4 endpoint returns 429/1113
# "Insufficient balance" for a coding-plan key. A pay-as-you-go (general) key:
# set ZAI_BASE_URL=https://api.z.ai/api/paas/v4/ in .env to override.
_ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4/"
_ZAI_TEMPERATURE = 0.6   # z.ai documented default; set explicitly for reproducibility
_ZAI_THINKING_ENABLED = False  # GLM-5.2 thinks by default: +5-15s/call AND it can
                               # burn the whole max_tokens budget on reasoning,
                               # returning EMPTY content. Off = fast + complete;
                               # flip True for a hard-reasoning tier if ever needed.

_MODEL_ALIASES = {  # CLI accepts aliases; the SDK needs real model ids
    # opus pinned to 4.7 (NOT the newest 4.8): same price ($5/$25 per 1M) and
    # 1M ctx, preferred here for the architect tier. NOTE the bare CLI alias
    # "opus" resolves to the CLI's latest (4.8), so configs pass explicit ids.
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


class AgentUnavailable(RuntimeError):
    """The agent backend is down for longer than any per-step backoff covers
    (spend cap, or a rate-limit window that outlasted the full retry budget).
    Propagated past the mutator so the campaign can stop instead of burning
    the remaining steps (run1 lost 14 steps, run2 lost 6 to this)."""


class SingleCoder:
    def __init__(self, backend="cli", model="opus", timeout_s=900, max_tokens=8000):
        self.backend = backend
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._zai_c = None          # cached openai client for the 'zai' backend
        self.call_counts = Counter()   # model -> calls (per-tier cost for MLflow)

    def digest(self, evidence: str) -> str:
        """No-op for a single coder: the one model reads the evidence directly.
        The tiered agent overrides this with a cheap analyst pre-pass."""
        return evidence

    def complete(self, prompt: str, tier=None) -> str:
        # tier is accepted (and ignored) so single<->tiered is a drop-in swap.
        self.call_counts[self.model] += 1
        if self.backend == "sdk":
            return self._sdk(prompt)
        if self.backend == "zai":
            return self._zai(prompt)
        return self._cli(prompt)

    def _cli(self, prompt):
        # Rate-limit windows last minutes-to-hours; without backoff one outage
        # burns every remaining step in seconds (run1 lost steps 16-29 that way).
        delay, last = 60, "claude CLI: no attempt made"
        for attempt in range(6):
            r = subprocess.run(
                ["claude", "-p", "--model", self.model],
                input=prompt, capture_output=True, text=True,
                timeout=self.timeout_s)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout
            err = (r.stderr.strip() or r.stdout.strip())[:500]
            last = f"claude CLI failed (rc={r.returncode}): {err}"
            # A monthly/spend hard cap will not clear within any backoff window;
            # retrying just sleeps 30min per step (run2 lost steps 16-21 to this).
            if "spend limit" in err.lower():
                raise AgentUnavailable(last)
            if attempt < 5:
                print(f"[agent] {last} -- retry {attempt + 2}/6 in {delay}s",
                      flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 900)
        raise AgentUnavailable(last)

    def _sdk(self, prompt):
        import anthropic
        client = anthropic.Anthropic()
        model = _MODEL_ALIASES.get(self.model, self.model)
        msg = client.messages.create(
            model=model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")

    def _zai_client(self):
        """Lazily build + cache the z.ai openai client. z.ai is OpenAI-compatible:
        same SDK, just base_url + key swapped. Cached for the agent's lifetime
        (one connection pool) -- unlike the _sdk path, which rebuilds per call."""
        if self._zai_c is None:
            from openai import OpenAI
            key = os.environ.get("ZAI_API_KEY") or os.environ.get("Z.AI_KEY")
            if not key:
                raise RuntimeError(
                    "zai backend selected but no ZAI_API_KEY / Z.AI_KEY in env "
                    "(put it in .env)")
            base = os.environ.get("ZAI_BASE_URL", _ZAI_BASE_URL)
            # max_retries=0: we do our own backoff below so a hard spend cap still
            # raises AgentUnavailable (matching _cli's contract), not a silent retry.
            self._zai_c = OpenAI(api_key=key, base_url=base,
                                 timeout=self.timeout_s, max_retries=0)
        return self._zai_c

    def _zai(self, prompt):
        # Same backoff + AgentUnavailable contract as _cli: a z.ai rate-limit
        # window lasts minutes, so without backoff one outage burns every
        # remaining step in seconds; a hard quota/spend cap never clears within
        # any backoff window, so we bail immediately instead of sleeping ~15min
        # per step (the run1/run2 trap _cli's comment warns about).
        client = self._zai_client()
        delay, last = 60, "zai: no attempt made"
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=self.model, max_tokens=self.max_tokens,
                    temperature=_ZAI_TEMPERATURE,
                    extra_body={"thinking": {"type":
                        "enabled" if _ZAI_THINKING_ENABLED else "disabled"}},
                    messages=[{"role": "user", "content": prompt}])
                content = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                err = f"{type(e).__name__}: {str(e).strip()}"[:500]
                low = err.lower()
                if any(s in low for s in ("quota", "insufficient", "spend limit",
                                          "exceeded your current", "balance",
                                          "no credit")):
                    raise AgentUnavailable(f"zai hard quota/spend: {err}")
                last = f"zai call failed (attempt {attempt + 1}/6): {err}"
                wait = delay
            else:
                if content:
                    return content
                last = f"zai returned empty content (attempt {attempt + 1}/6)"
                wait = min(delay, 30)   # empty != rate-limited; don't sleep minutes
            if attempt < 5:
                print(f"[agent] {last} -- retry in {wait}s", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 900)
        raise AgentUnavailable(last)
