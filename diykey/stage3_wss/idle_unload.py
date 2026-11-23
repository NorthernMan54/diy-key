# SPDX-License-Identifier: AGPL-3.0-or-later
"""Release big, quick-to-reload models after they sit unused (DIYKEY_IDLE_UNLOAD_MIN).

Why: on a 16 GB Mac the worker holds ~5.6 GB, 3.2 GB of it GPU buffers (footprint,
2026-11-23). macOS compresses or swaps most of that while idle, but it still fills the
compressor and swap that other apps need. Only models that reload in seconds are unloaded:
the VLM (1.4 GB, ~4 s) and GFPGAN face enhance (0.9 GB, ~3 s). Measured warm-ups
2026-11-23: RAM++ 82-337 s and CLIP 6-31 s, so those stay loaded. Protect's limits leave
room for the quick reloads: a tagging task (whose summary the VLM writes) waits 30 min,
on-demand summary requestAi 15 s (M/630689:25, M/490990:11-24).

Each module reports its own last use and unloads only under its inference lock taken
without blocking, so a model in use is never released. The next call loads it again.
"""
import gc
import logging
import os
import sys
import threading
import time

log = logging.getLogger("diykey.idle_unload")

CHECK_S = 60


def idle_seconds(env=None):
    """DIYKEY_IDLE_UNLOAD_MIN in seconds; 0 (off) when unset, non-positive or bad."""
    raw = (env if env is not None else os.environ).get("DIYKEY_IDLE_UNLOAD_MIN", "")
    try:
        minutes = float(raw)
    except ValueError:
        if raw.strip():
            log.warning("DIYKEY_IDLE_UNLOAD_MIN=%r is not a number; idle unload off", raw)
        return 0
    return int(minutes * 60) if minutes > 0 else 0


def free_accelerator_caches():
    """Hand freed GPU buffers back to the OS. Dropping the model frees Python objects; MLX
    and PyTorch MPS keep their own buffer caches until told to clear them."""
    gc.collect()
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
        except Exception:  # noqa: BLE001 -- best effort; the model is already dropped
            log.debug("mlx clear_cache failed", exc_info=True)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            log.debug("torch mps empty_cache failed", exc_info=True)


def sweep(unloaders, idle_s, now=None, footprint=None):
    """Run each (name, fn(idle_s, now) -> bool) once; free caches and log when any unloaded.
    Returns the names unloaded."""
    now = time.monotonic() if now is None else now
    done = []
    for name, fn in unloaders:
        try:
            if fn(idle_s, now):
                done.append(name)
        except Exception:  # noqa: BLE001 -- never let the sweep kill the thread
            log.exception("idle unload of %s failed", name)
    if done:
        before = footprint() if footprint else None
        free_accelerator_caches()
        after = footprint() if footprint else None
        note = f" (process {before:.0f} -> {after:.0f} MB)" if before and after else ""
        log.info("idle %d min: unloaded %s%s", idle_s // 60, ", ".join(done), note)
    return done


def start(unloaders, footprint=None, idle_s=None):
    """Start the sweep thread when DIYKEY_IDLE_UNLOAD_MIN is set; returns it or None."""
    idle_s = idle_seconds() if idle_s is None else idle_s
    if not idle_s:
        return None

    def loop():
        while True:
            time.sleep(CHECK_S)
            sweep(unloaders, idle_s, footprint=footprint)

    log.info("idle unload: %s after %d min unused", ", ".join(n for n, _ in unloaders),
             idle_s // 60)
    t = threading.Thread(target=loop, daemon=True, name="diykey-idle-unload")
    t.start()
    return t
