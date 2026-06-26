"""Offline tests for the z.ai (GLM) agent backend + its config wiring.

Covers:
  * config.py: _autoswitch_backend precedence (an explicit backend is never
    clobbered by a key; z.ai key -> "zai"; anthropic key -> "sdk"; nothing ->
    None), and the glm-5.2 model default for leftover claude-* slugs once the
    backend resolves to zai.
  * agents/single.py: SingleCoder(backend="zai") dispatches to _zai, builds an
    openai client pointed at z.ai, and passes model=glm-5.2 through -- all via a
    stubbed openai client (no network, no real key).

Run: PYTHONPATH=$PWD/src .venv/bin/python -m tests.test_zai_backend
"""
import os

from graphretr_opt.agents.single import SingleCoder
from graphretr_opt.config import _autoswitch_backend, load_config

_PROVIDER_KEYS = ("ZAI_API_KEY", "Z.AI_KEY", "ANTHROPIC_API_KEY")


def _check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"ok  {name}")


def _with_keys(zai=None, anthropic=None):
    """Clear the provider keys, optionally set zai/anthropic, return a restore
    callable. Keeps the auto-switch tests independent of .env / shell state."""
    snapshot = {k: os.environ.get(k) for k in _PROVIDER_KEYS}
    for k in _PROVIDER_KEYS:
        os.environ.pop(k, None)
    if zai:
        os.environ["ZAI_API_KEY"] = zai
    if anthropic:
        os.environ["ANTHROPIC_API_KEY"] = anthropic

    def restore():
        for k, v in snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return restore


# ---- _autoswitch_backend (pure; key presence only) --------------------------

def test_autoswitch_no_keys_no_backend():
    restore = _with_keys()
    try:
        _check("no keys, nothing specified -> None",
               _autoswitch_backend({}, {}, None) is None)
    finally:
        restore()


def test_autoswitch_zai_key():
    restore = _with_keys(zai="k")
    try:
        _check("zai key, nothing specified -> zai",
               _autoswitch_backend({}, {}, None) == "zai")
    finally:
        restore()


def test_autoswitch_anthropic_key():
    restore = _with_keys(anthropic="k")
    try:
        _check("anthropic key only -> sdk",
               _autoswitch_backend({}, {}, None) == "sdk")
    finally:
        restore()


def test_autoswitch_explicit_not_clobbered():
    # A backend specified in yaml / env / kwargs is NEVER overridden by a key.
    restore = _with_keys(zai="k", anthropic="k")
    try:
        _check("yaml backend wins over keys",
               _autoswitch_backend({"mutator_backend": "cli"}, {}, None) is None)
        _check("env backend wins over keys",
               _autoswitch_backend({}, {}, "cli") is None)
        _check("kwarg backend wins over keys",
               _autoswitch_backend({}, {"mutator_backend": "cli"}, None) is None)
    finally:
        restore()


# ---- load_config model defaulting -------------------------------------------

def test_zai_defaults_claude_slug_to_glm():
    cfg = load_config(mutator_backend="zai", mutator_model="claude-opus-4-7")
    _check("zai + claude slug -> glm-5.2", cfg.mutator_model == "glm-5.2")


def test_zai_respects_explicit_glm_slug():
    cfg = load_config(mutator_backend="zai", mutator_model="glm-4.6")
    _check("zai + explicit glm slug respected", cfg.mutator_model == "glm-4.6")


def test_live_config_resolves_zai_glm():
    # The real campaign.yaml + .env: the optimizer should be on zai / glm-5.2.
    cfg = load_config()
    _check("campaign backend is zai", cfg.mutator_backend == "zai")
    _check("campaign mutator model is glm-5.2", cfg.mutator_model == "glm-5.2")


# ---- SingleCoder zai dispatch (stubbed openai client) -----------------------

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, owner):
        self._o = owner

    def create(self, **kw):
        self._o.create_kw = kw
        return _Resp("REPLY")


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class _FakeOpenAI:
    """Stand-in for openai.OpenAI: records init + create kwargs and returns a
    canned message. No network."""

    def __init__(self, **kw):
        self.init_kw = kw
        self.create_kw = {}
        self.chat = _Chat(self)


def test_zai_dispatch_and_passthrough():
    import openai
    real = openai.OpenAI
    openai.OpenAI = _FakeOpenAI
    os.environ["ZAI_API_KEY"] = "test-key"
    try:
        ag = SingleCoder("zai", "glm-5.2", timeout_s=30)
        out = ag.complete("hello world")
    finally:
        openai.OpenAI = real
        os.environ.pop("ZAI_API_KEY", None)
    client = ag._zai_c
    _check("zai returns message text", out == "REPLY")
    _check("zai client built (cached)", client is not None)
    _check("zai client base_url is z.ai",
           "z.ai" in client.init_kw.get("base_url", ""))
    _check("zai create model is glm-5.2", client.create_kw.get("model") == "glm-5.2")
    _check("zai create prompt passed through",
           client.create_kw["messages"][0]["content"] == "hello world")
    _check("zai create temperature set",
           client.create_kw.get("temperature") == 0.6)


def test_zai_missing_key_raises():
    restore = _with_keys()           # no provider keys at all
    try:
        ag = SingleCoder("zai", "glm-5.2", timeout_s=5)
        try:
            ag.complete("hi")
            _check("zai missing key raises", False)
        except RuntimeError as e:
            _check("zai missing key raises RuntimeError", "ZAI_API_KEY" in str(e))
    finally:
        restore()


def main():
    test_autoswitch_no_keys_no_backend()
    test_autoswitch_zai_key()
    test_autoswitch_anthropic_key()
    test_autoswitch_explicit_not_clobbered()
    test_zai_defaults_claude_slug_to_glm()
    test_zai_respects_explicit_glm_slug()
    test_live_config_resolves_zai_glm()
    test_zai_dispatch_and_passthrough()
    test_zai_missing_key_raises()
    print("\nall zai backend tests passed")


if __name__ == "__main__":
    main()
