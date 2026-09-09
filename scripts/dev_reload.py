"""Watch ``src/`` and restart Cove Video Editor when Python files change."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from watchfiles import watch
from watchfiles.filters import PythonFilter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FFMPEG = ROOT / "ffmpeg"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    # Prepend so a stale/empty PYTHONPATH from the debugger cannot hide src/.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(SRC) if not existing else f"{SRC}{os.pathsep}{existing}"
    )
    if FFMPEG.is_dir():
        env["PATH"] = f"{FFMPEG}{os.pathsep}{env.get('PATH', '')}"
    # Avoid the child re-attaching to the parent's debugpy session.
    for key in list(env):
        if key.upper().startswith("PYDEVD_") or key.upper().startswith("DEBUGPY_"):
            env.pop(key, None)
    env.pop("PYTHONSTARTUP", None)
    return env


def _start() -> subprocess.Popen[str]:
    # Inject src onto sys.path in-process so this works even when the
    # debugger/shell fails to pass PYTHONPATH through to the child.
    boot = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "runpy.run_module('cove_video_editor', run_name='__main__')"
    )
    print(f"[dev-reload] starting {PYTHON} -m cove_video_editor", flush=True)
    return subprocess.Popen(
        [str(PYTHON), "-c", boot],
        cwd=str(ROOT),
        env=_env(),
    )


def _stop(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    else:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def main() -> int:
    if not SRC.is_dir():
        print(f"[dev-reload] missing src dir: {SRC}", file=sys.stderr)
        return 1

    proc = _start()
    print(f"[dev-reload] watching {SRC} — save a .py file to refresh", flush=True)
    try:
        for changes in watch(SRC, watch_filter=PythonFilter(), debounce=400):
            names = ", ".join(sorted({Path(p).name for _, p in changes}))
            print(f"[dev-reload] changed: {names} — restarting…", flush=True)
            _stop(proc)
            time.sleep(0.3)
            proc = _start()
    except KeyboardInterrupt:
        print("\n[dev-reload] stopped", flush=True)
    finally:
        _stop(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
