"""Session-based ephemeral index manager.

Automatically deletes per-session FAISS index directories after a period of inactivity.
Because Streamlit does not expose a per-session explicit 'on disconnect' event in the
public API, we approximate session end by an inactivity TTL.
"""
from __future__ import annotations
import threading
import time
from typing import Callable, Dict, Tuple

TTL_SECONDS = 120  # delete after 2 minutes of no activity
_CLEAN_INTERVAL = 30  # how often background janitor runs

# session_id -> (last_touch_ts, cleanup_fn)
_registry: Dict[str, Tuple[float, Callable[[], None]]] = {}
_lock = threading.Lock()
_started = False

def register(session_id: str, cleanup_fn: Callable[[], None]):
    with _lock:
        _registry[session_id] = (time.time(), cleanup_fn)


def touch(session_id: str):
    with _lock:
        if session_id in _registry:
            _, fn = _registry[session_id]
            _registry[session_id] = (time.time(), fn)


def _janitor_loop():  # pragma: no cover
    while True:
        time.sleep(_CLEAN_INTERVAL)
        now = time.time()
        stale: Dict[str, Callable[[], None]] = {}
        with _lock:
            for sid, (ts, fn) in list(_registry.items()):
                if now - ts > TTL_SECONDS:
                    stale[sid] = fn
                    del _registry[sid]
        for sid, fn in stale.items():
            try:
                fn()
            except Exception:
                pass


def ensure_started():
    global _started
    if _started:
        return
    with _lock:
        if _started:
            return
        t = threading.Thread(target=_janitor_loop, name="session-janitor", daemon=True)
        t.start()
        _started = True
