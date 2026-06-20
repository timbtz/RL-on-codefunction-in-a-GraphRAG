"""atomic_io -- crash-safe JSON/text writes (temp file + os.replace).

The persist idiom across the optimizer (`_ScoreCache`, `RejectedBuffer`, and the
Phase-1 campaign checkpoint) previously truncated-then-wrote in place
(`open(path, "w"); json.dump(...)`), which leaves a half-written or empty file if
the process dies mid-dump. `os.replace` is atomic on POSIX, so a reader either
sees the old complete file or the new complete file -- never a torn one. This is
the explicit "do NOT inherit openEvolve's non-atomic writes" fix.
"""
import json
import os


def atomic_write_text(path: str, text: str):
    """Write `text` to `path` atomically (write tmp in the same dir, fsync, rename)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_json(path: str, obj, indent: int = 1):
    """JSON-dump `obj` to `path` atomically."""
    atomic_write_text(path, json.dumps(obj, indent=indent))
