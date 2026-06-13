"""SingleCoder -- one reflective coding agent. Diagnoses misses and emits a
new program in a single combined call (SkillOpt style, reasoning model).

Backend 'cli': `claude -p --model <m>` subprocess -- zero config, works on a
box with only the authenticated CLI. Backend 'sdk': the anthropic client when
an API key is present (cleaner transcripts, token counts).
"""
import subprocess
import time
from collections import Counter

_MODEL_ALIASES = {  # CLI accepts aliases; the SDK needs real model ids
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


class AgentUnavailable(RuntimeError):
    """The agent backend is down for longer than any per-step backoff covers
    (spend cap, or a rate-limit window that outlasted the full retry budget).
    Propagated past the mutator so the campaign can stop instead of burning
    the remaining steps (run1 lost 14 steps, run2 lost 6 to this)."""


class SingleCoder:
    def __init__(self, backend="cli", model="opus", timeout_s=900):
        self.backend = backend
        self.model = model
        self.timeout_s = timeout_s
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
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")
