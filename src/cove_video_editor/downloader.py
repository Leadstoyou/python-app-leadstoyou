from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, QStandardPaths, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

_PROGRESS_RE = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
_QUOTED_MP4_RE = re.compile(r'"([^"]+\.mp4)"')

# (yt-dlp lang selector, checkbox label). Prefer official + auto captions.
SUBTITLE_LANG_OPTIONS: list[tuple[str, str]] = [
    ("zh.*", "中文 (Chinese)"),
    ("en.*", "English"),
    ("vi.*", "Tiếng Việt"),
]


def _quality_format(max_height: int | None = None) -> str:
    height = f"[height<={max_height}]" if max_height is not None else ""
    return "/".join(
        (
            f"bestvideo[ext=mp4][vcodec^=avc1]{height}+bestaudio[ext=m4a][acodec^=mp4a]",
            f"bestvideo[ext=mp4][vcodec^=avc1]{height}+bestaudio[ext=m4a]",
            f"bestvideo[ext=mp4]{height}+bestaudio[ext=m4a]",
            f"best[ext=mp4][vcodec^=avc1][acodec^=mp4a]{height}",
            f"best[ext=mp4]{height}",
            f"bestvideo{height}+bestaudio",
            f"best{height}",
        ),
    )


QUALITY_FORMATS = {
    "Best": _quality_format(),
    "1080p": _quality_format(1080),
    "720p": _quality_format(720),
    "480p": _quality_format(480),
    "360p": _quality_format(360),
}


def _yt_dlp_command() -> list[str] | None:
    exe_name = "yt-dlp.exe" if os.name == "nt" else "yt-dlp"
    for directory in _yt_dlp_dirs():
        candidate = directory / exe_name
        if candidate.is_file():
            return [str(candidate)]

    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    return None


def _yt_dlp_dirs() -> list[Path]:
    dirs: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass))
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent)
    dirs.append(Path(__file__).resolve().parent.parent.parent / "assets" / "bin")
    return dirs


class DownloadWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(Path, list)  # video path, related .srt paths
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        url: str,
        quality: str,
        output_folder: Path,
        ffmpeg_path: str | None = None,
        subtitle_langs: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._url = url
        self._quality = quality
        self._output_folder = output_folder
        self._ffmpeg_path = ffmpeg_path
        self._subtitle_langs = list(subtitle_langs or [])
        self._proc: subprocess.Popen[str] | None = None
        self._cancelled = False

    @Slot()
    def run(self) -> None:
        command = _yt_dlp_command()
        if command is None:
            self.failed.emit(
                "yt-dlp is required to download videos. Install yt-dlp and try again.",
            )
            return

        self._output_folder.mkdir(parents=True, exist_ok=True)
        fmt = QUALITY_FORMATS.get(self._quality, QUALITY_FORMATS["Best"])
        cmd = [
            *command,
            "--newline",
            "--no-playlist",
            "--merge-output-format", "mp4",
            "--remux-video", "mp4",
            "--format-sort", "vcodec:h264,acodec:aac,ext:mp4:m4a",
            "-f", fmt,
            "-P", str(self._output_folder),
            "-o", "%(title).200B [%(id)s].%(ext)s",
            "--print", "after_move:filepath",
        ]
        if self._subtitle_langs:
            cmd.extend(
                [
                    "--write-subs",
                    "--write-auto-subs",
                    "--sub-langs",
                    ",".join(self._subtitle_langs),
                    "--convert-subs",
                    "srt",
                ],
            )
        if self._ffmpeg_path:
            cmd.extend(["--ffmpeg-location", self._ffmpeg_path])
        cmd.append(self._url)

        self.status.emit("Starting download...")
        out_path: Path | None = None
        last_pct = 0
        recent_output: deque[str] = deque(maxlen=8)
        error_output: deque[str] = deque(maxlen=4)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert self._proc.stdout is not None
            for raw_line in self._proc.stdout:
                if self._cancelled:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                pct = self._progress_percent(line)
                if pct is not None:
                    last_pct = max(last_pct, pct)
                    self.progress.emit(last_pct)
                    self.status.emit("Downloading...")
                    continue
                maybe_path = self._path_from_line(line)
                if maybe_path is not None:
                    out_path = maybe_path
                if self._is_warning_line(line):
                    continue
                if self._is_error_line(line):
                    error_output.append(line)
                    recent_output.append(line)
                    self.status.emit(line)
                    continue
                recent_output.append(line)
                status = self._status_from_output(line)
                if status is not None:
                    self.status.emit(status)

            if self._cancelled:
                self._finish_cancelled_process()
                self.cancelled.emit()
                return
            rc = self._proc.wait()
            if rc != 0:
                self.failed.emit(self._failure_message(rc, error_output, recent_output))
                return
            if out_path is None or not out_path.exists():
                out_path = self._newest_mp4()
            if out_path is None or not out_path.exists():
                self.failed.emit("Download finished, but no MP4 file was found.")
                return
            srt_paths = self._related_srt_files(out_path) if self._subtitle_langs else []
            self.progress.emit(100)
            if srt_paths:
                self.status.emit(
                    f"Downloaded {out_path.name} + {len(srt_paths)} SRT file(s)",
                )
            else:
                self.status.emit(f"Downloaded {out_path.name}")
            self.finished.emit(out_path, srt_paths)
        except FileNotFoundError:
            self.failed.emit(
                "yt-dlp is required to download videos. Install yt-dlp and try again.",
            )
        except Exception as exc:  # noqa: BLE001
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
        finally:
            self._proc = None

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _finish_cancelled_process(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

    def _newest_mp4(self) -> Path | None:
        mp4s = list(self._output_folder.glob("*.mp4"))
        if not mp4s:
            return None
        return max(mp4s, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _related_srt_files(video: Path) -> list[Path]:
        """SRT files yt-dlp wrote next to ``video`` (same stem / stem.lang)."""
        stem = video.stem
        parent = video.parent
        found: list[Path] = []
        for pattern in (f"{stem}.srt", f"{stem}.*.srt"):
            found.extend(p for p in parent.glob(pattern) if p.is_file())
        # Prefer unique paths, newest language variants first by name.
        unique = sorted({p.resolve() for p in found}, key=lambda p: p.name.lower())
        return unique

    @staticmethod
    def _progress_percent(line: str) -> int | None:
        match = _PROGRESS_RE.search(line)
        if not match:
            return None
        return max(0, min(100, int(float(match.group(1)))))

    @staticmethod
    def _is_warning_line(line: str) -> bool:
        return line.startswith("WARNING:")

    @staticmethod
    def _is_error_line(line: str) -> bool:
        return line.startswith("ERROR:")

    @staticmethod
    def _status_from_output(line: str) -> str | None:
        lowered = line.lower()
        if "[merger]" in lowered or "merging formats" in lowered:
            return "Merging..."
        if (
            "remux" in lowered
            or "post-process" in lowered
            or "postprocess" in lowered
            or "[movefiles]" in lowered
            or "deleting original file" in lowered
            or "converting" in lowered
        ):
            return "Finalizing..."
        if "subtitle" in lowered or "[info] writing" in lowered:
            return "Downloading subtitles..."
        if line.startswith("[download]"):
            return "Downloading..."
        return None

    @staticmethod
    def _failure_message(
        return_code: int,
        error_output: deque[str],
        recent_output: deque[str],
    ) -> str:
        message = f"yt-dlp exited with status {return_code}."
        context = list(dict.fromkeys([*error_output, *recent_output]))
        if context:
            message = f"{message}\n\n" + "\n".join(context)
        return message

    @staticmethod
    def _path_from_line(line: str) -> Path | None:
        candidate = Path(line.strip('"'))
        if candidate.suffix.lower() == ".mp4" and candidate.exists():
            return candidate
        quoted = _QUOTED_MP4_RE.search(line)
        if quoted:
            candidate = Path(quoted.group(1))
            if candidate.exists():
                return candidate
        return None


class DownloadVideoDialog(QDialog):
    downloadFinished = Signal(Path, list)  # video path, related .srt paths

    def __init__(self, parent=None, ffmpeg_path: str | None = None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Download Video")
        self.setModal(False)
        self.resize(520, 300)
        self._ffmpeg_path = ffmpeg_path
        self._thread: QThread | None = None
        self._worker: DownloadWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://...")
        form.addRow("URL", self.url_edit)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(list(QUALITY_FORMATS.keys()))
        form.addRow("Quality", self.quality_combo)

        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(str(self._default_output_folder()))
        self.folder_edit.setReadOnly(True)
        self.folder_btn = QPushButton("Choose...")
        self.folder_btn.clicked.connect(self._choose_folder)
        folder_row.addWidget(self.folder_edit, stretch=1)
        folder_row.addWidget(self.folder_btn)
        form.addRow("Output folder", folder_row)

        subs_row = QHBoxLayout()
        self.sub_checkboxes: dict[str, QCheckBox] = {}
        for lang_code, label in SUBTITLE_LANG_OPTIONS:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setToolTip(
                "Download official and auto-generated captions as .srt "
                "(when the site provides them).",
            )
            self.sub_checkboxes[lang_code] = cb
            subs_row.addWidget(cb)
        subs_row.addStretch(1)
        form.addRow("SRT subtitles", subs_row)
        root.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("Muted")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_download)
        self.download_btn = QPushButton("Download")
        self.download_btn.setObjectName("PrimaryButton")
        self.download_btn.clicked.connect(self._start_download)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.download_btn)
        root.addLayout(btn_row)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self.cancel_and_wait()
        super().closeEvent(event)

    def cancel_and_wait(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def _default_output_folder(self) -> Path:
        location = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
        if location:
            return Path(location)
        return Path.home() / "Downloads"

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Output Folder",
            self.folder_edit.text(),
        )
        if folder:
            self.folder_edit.setText(folder)

    def _start_download(self) -> None:
        url = self.url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "URL required", "Paste a video URL to download.")
            return
        output_folder = Path(self.folder_edit.text()).expanduser()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting download...")
        self.download_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.url_edit.setEnabled(False)
        self.quality_combo.setEnabled(False)
        self.folder_btn.setEnabled(False)
        for cb in self.sub_checkboxes.values():
            cb.setEnabled(False)

        thread = QThread(self)
        worker = DownloadWorker(
            url,
            self.quality_combo.currentText(),
            output_folder,
            self._ffmpeg_path,
            subtitle_langs=self._selected_subtitle_langs(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        worker.status.connect(self.status_label.setText, Qt.QueuedConnection)
        worker.finished.connect(self._download_finished, Qt.QueuedConnection)
        worker.failed.connect(self._download_failed, Qt.QueuedConnection)
        worker.cancelled.connect(self._download_cancelled, Qt.QueuedConnection)
        worker.finished.connect(thread.quit, Qt.QueuedConnection)
        worker.failed.connect(thread.quit, Qt.QueuedConnection)
        worker.cancelled.connect(thread.quit, Qt.QueuedConnection)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _selected_subtitle_langs(self) -> list[str]:
        return [
            code
            for code, cb in self.sub_checkboxes.items()
            if cb.isChecked()
        ]

    def _cancel_download(self) -> None:
        if self._worker is not None:
            self.status_label.setText("Cancelling...")
            self._worker.cancel()

    def _download_finished(self, path: Path, srt_paths: list) -> None:
        self.downloadFinished.emit(path, list(srt_paths or []))
        self.accept()

    def _download_failed(self, message: str) -> None:
        self.status_label.setText(message)
        QMessageBox.warning(self, "Download failed", message)
        self._reset_controls()

    def _download_cancelled(self) -> None:
        self.status_label.setText("Cancelled")
        self._reset_controls()

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _reset_controls(self) -> None:
        self.download_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.url_edit.setEnabled(True)
        self.quality_combo.setEnabled(True)
        self.folder_btn.setEnabled(True)
        for cb in self.sub_checkboxes.values():
            cb.setEnabled(True)
