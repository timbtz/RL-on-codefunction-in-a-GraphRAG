"""SingleCoder -- one reflective coding agent. Diagnoses misses and emits a
new program in a single combined call (SkillOpt style, reasoning model).

Backend 'cli': `claude -p --model <m>` subprocess -- zero config, works on a
box with only the authenticated CLI. Backend 'sdk': the anthropic client when
an API key is present (cleaner transcripts, token counts).
"""
import subprocess

_MODEL_ALIASES = {  # CLI accepts aliases; the SDK needs real model ids
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


class SingleCoder:
    def __init__(self, backend="cli", model="opus", timeout_s=900):
        self.backend = backend
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, prompt: str) -> str:
        if self.backend == "sdk":
            return self._sdk(prompt)
        return self._cli(prompt)

    def _cli(self, prompt):
        r = subprocess.run(
            ["claude", "-p", "--model", self.model],
            input=prompt, capture_output=True, text=True, timeout=self.timeout_s)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError(
                f"claude CLI failed (rc={r.returncode}): {r.stderr.strip()[:500]}")
        return r.stdout

    def _sdk(self, prompt):
        import anthropic
        client = anthropic.Anthropic()
        model = _MODEL_ALIASES.get(self.model, self.model)
        msg = client.messages.create(
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")
