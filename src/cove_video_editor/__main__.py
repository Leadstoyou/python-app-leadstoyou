import faulthandler
import os
import sys

# PyInstaller `--windowed` builds detach from the console, which sets
# `sys.stderr` to None. `faulthandler.enable()` without args tries to
# register stderr's fd and raises `RuntimeError: sys.stderr is None`
# before the GUI even starts. Point it at a real file when there's no
# stderr, so crash tracebacks still land somewhere (a log next to the
# user data dir) instead of nuking the app on startup.
if sys.stderr is not None and hasattr(sys.stderr, "fileno"):
    try:
        faulthandler.enable()
    except (RuntimeError, OSError, ValueError):
        pass
else:
    try:
        from .portable import is_portable, portable_data_dir
        if is_portable():
            log_dir = portable_data_dir("cove-video-editor")
        elif sys.platform == "win32":
            log_dir = os.path.join(
                os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                "CoveVideoEditor",
            )
        else:
            log_dir = os.path.join(os.path.expanduser("~"), ".cove-video-editor")
        os.makedirs(log_dir, exist_ok=True)
        _fault_log = open(os.path.join(log_dir, "faulthandler.log"), "a", buffering=1)
        faulthandler.enable(file=_fault_log)
    except (OSError, RuntimeError, ValueError):
        pass

# Qt's default media backend on Linux is GStreamer, which crashes on AV1 /
# unusual codecs. PySide6 6.5+ ships an FFmpeg-based backend that covers
# everything we need — opt into it before QApplication touches the plugins.
os.environ.setdefault("QT_MEDIA_BACKEND", "ffmpeg")


def _model_cache_dir() -> str:
    """Where faster-whisper and FunASR/ModelScope should cache downloads.
    All default to the user's home directory (``~/.cache``), which on
    Windows is almost always the C: drive regardless of where the app
    itself lives. Keep model downloads next to the app instead — same
    drive as a source checkout, or next to the exe / portable data dir for
    a built release — so several GB of speech models don't quietly land on
    a drive the user didn't choose.
    """
    from .portable import is_portable, portable_data_dir

    if getattr(sys, "frozen", False):
        if is_portable():
            base = portable_data_dir("cove-video-editor")
        elif sys.platform == "win32":
            base = os.path.join(
                os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                "CoveVideoEditor",
            )
        else:
            base = os.path.join(os.path.expanduser("~"), ".cove-video-editor")
    else:
        base = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        base = os.path.join(base, "assets")
    return os.path.join(base, "models")


_models_dir = _model_cache_dir()
os.makedirs(_models_dir, exist_ok=True)
# faster-whisper downloads its model via huggingface_hub.
os.environ.setdefault("HF_HOME", os.path.join(_models_dir, "huggingface"))
# FunASR (Chinese speech-to-text) downloads its models via ModelScope.
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(_models_dir, "modelscope"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from . import theme  # noqa: E402
from .app import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Cove Video Editor")
    app.setOrganizationName("Cove")
    theme.apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
