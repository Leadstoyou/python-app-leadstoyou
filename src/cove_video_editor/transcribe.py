"""Generate SRT subtitles from timeline video audio via faster-whisper."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from . import ffmpeg_utils as ff
from .clip import Clip, sort_clips
from .exporter import _format_srt_ts

if os.name == "nt":
    _CREATE_NO_WINDOW = 0x08000000
    _POPEN_KWARGS: dict = {"creationflags": _CREATE_NO_WINDOW}
else:
    _POPEN_KWARGS = {}

# Format combo label → Whisper language code.
SUBTITLE_LANG_FORMATS: dict[str, str] = {
    "English (.srt)": "en",
    "中文 Chinese (.srt)": "zh",
    "Tiếng Việt (.srt)": "vi",
}

# "Translate to" combo label → Argos Translate target language code.
# ``None`` means keep the transcript in the spoken (source) language.
TRANSLATE_LANG_FORMATS: dict[str, str | None] = {
    "None (keep original)": None,
    "English": "en",
    "中文 Chinese": "zh",
    "Tiếng Việt": "vi",
}

_MODEL_NAME = "base"


def cues_to_srt(cues: list[tuple[float, float, str]]) -> str:
    blocks: list[str] = []
    for i, (start, end, text) in enumerate(cues, start=1):
        body = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not body or end <= start:
            continue
        blocks.append(
            f"{i}\n{_format_srt_ts(start)} --> {_format_srt_ts(end)}\n{body}\n"
        )
    return "\n".join(blocks)


def extract_timeline_wav(clips: list[Clip], wav_path: Path) -> None:
    """Render timeline clip audio to 16 kHz mono WAV for Whisper."""
    ordered = sort_clips(clips)
    if not ordered:
        raise RuntimeError("No clips on the timeline to transcribe.")

    ffmpeg = ff.require_ffmpeg()
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: one video/audio clip with speech.
    if len(ordered) == 1:
        clip = ordered[0]
        if clip.asset.kind == "image" or not clip.asset.has_audio:
            raise RuntimeError(
                "The timeline clip has no audio to transcribe. "
                "Add a video (or audio) clip with speech."
            )
        duration = max(0.05, clip.src_span / max(clip.speed, 0.01))
        cmd = [
            ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{clip.src_start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(clip.path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(wav_path),
        ]
        subprocess.run(cmd, check=True, **_POPEN_KWARGS)
        return

    # Multi-clip: trim each source, then concat (silence for image / no-audio).
    cmd: list[str] = [ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error"]
    filter_parts: list[str] = []
    concat_labels: list[str] = []
    input_index = 0

    for i, clip in enumerate(ordered):
        dur = max(0.05, clip.timeline_length)
        label = f"a{i}"
        if clip.asset.kind == "image" or not clip.asset.has_audio:
            filter_parts.append(
                f"anullsrc=channel_layout=mono:sample_rate=16000:d={dur:.3f}[{label}]"
            )
        else:
            cmd.extend(["-i", str(clip.path)])
            src_dur = max(0.05, clip.src_span)
            filter_parts.append(
                f"[{input_index}:a:0]"
                f"atrim=start={clip.src_start:.3f}:duration={src_dur:.3f},"
                f"asetpts=PTS-STARTPTS,aresample=16000,aformat=channel_layouts=mono"
                f"[{label}]"
            )
            input_index += 1
        concat_labels.append(f"[{label}]")

    filter_parts.append(
        f"{''.join(concat_labels)}concat=n={len(ordered)}:v=0:a=1[outa]"
    )
    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outa]",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav_path),
    ])
    subprocess.run(cmd, check=True, **_POPEN_KWARGS)


def transcribe_wav(wav_path: Path, language: str) -> list[tuple[float, float, str]]:
    """Run faster-whisper and return ``(start, end, text)`` cues."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is not installed. Run:\n"
            "  pip install faster-whisper"
        ) from exc

    model = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    cues: list[tuple[float, float, str]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        cues.append((float(seg.start), float(seg.end), text))
    return cues


def _argos_ensure_package(
    from_code: str, to_code: str, status_cb=None,  # noqa: ANN001
) -> None:
    """Download + install the Argos Translate model for one language pair.

    A no-op once installed. Needs internet the first time a given pair is
    used; the model is cached locally afterward (fully offline from then on).
    """
    import argostranslate.package as apackage

    installed = apackage.get_installed_packages()
    if any(p.from_code == from_code and p.to_code == to_code for p in installed):
        return

    if status_cb:
        status_cb(
            f"Downloading translation model {from_code}→{to_code} "
            "(first time only)…"
        )
    apackage.update_package_index()
    available = apackage.get_available_packages()
    match = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if match is None:
        raise RuntimeError(
            f"No Argos Translate model available for {from_code} → {to_code}."
        )
    apackage.install_from_path(match.download())


def _argos_get_translation(from_code: str, to_code: str, status_cb=None):  # noqa: ANN001
    """Return a ready-to-use ``ITranslation`` for ``from_code`` → ``to_code``.

    Installs the direct model if available; otherwise pivots through English
    (installing both legs), since Argos's package index doesn't cover every
    pair directly (e.g. zh → vi).
    """
    import argostranslate.translate as atranslate

    def _lang(code: str):
        lang = next(
            (l for l in atranslate.get_installed_languages() if l.code == code), None
        )
        if lang is None:
            raise RuntimeError(f"Argos Translate has no installed language '{code}'.")
        return lang

    if from_code == to_code:
        raise RuntimeError("Source and target languages are the same.")

    direct_error: RuntimeError | None = None
    try:
        _argos_ensure_package(from_code, to_code, status_cb)
        direct = _lang(from_code).get_translation(_lang(to_code))
        if direct is not None:
            return direct
    except RuntimeError as exc:
        direct_error = exc

    if "en" in (from_code, to_code):
        raise direct_error or RuntimeError(
            f"No Argos Translate model for {from_code} → {to_code}."
        )
    _argos_ensure_package(from_code, "en", status_cb)
    _argos_ensure_package("en", to_code, status_cb)
    first = _lang(from_code).get_translation(_lang("en"))
    second = _lang("en").get_translation(_lang(to_code))
    if first is None or second is None:
        raise RuntimeError(
            f"Could not build a translation path {from_code} → en → {to_code}."
        )

    class _Pivoted:
        def translate(self, text: str) -> str:
            return second.translate(first.translate(text))

    return _Pivoted()


def translate_cues(
    cues: list[tuple[float, float, str]],
    from_lang: str,
    target_lang: str,
    status_cb=None,  # noqa: ANN001
    progress_cb=None,  # noqa: ANN001
) -> list[tuple[float, float, str]]:
    """Translate cue text from ``from_lang`` to ``target_lang``, offline, via
    Argos Translate. Keeps original timing.

    ``status_cb(msg)`` reports coarse phase changes (model download vs.
    translating); ``progress_cb(done, total)`` reports per-cue progress —
    both matter here because a first-time model download plus per-line CPU
    inference can take long enough that a static "70%" reads as a hang.
    """
    try:
        import argostranslate.translate  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "argostranslate is not installed. Run:\n"
            "  pip install argostranslate"
        ) from exc

    try:
        translation = _argos_get_translation(from_lang, target_lang, status_cb)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Translation setup failed: {exc}") from exc

    if status_cb:
        status_cb("Translating…")

    total = len(cues)
    out: list[tuple[float, float, str]] = []
    for i, (start, end, text) in enumerate(cues):
        try:
            new_text = translation.translate(text).strip()
        except Exception:  # noqa: BLE001
            new_text = text
        out.append((start, end, new_text or text))
        if progress_cb:
            progress_cb(i + 1, total)
    return out


class TranscribeWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(Path)
    failed = Signal(str)

    def __init__(
        self,
        clips: list[Clip],
        output: Path,
        language: str,
        translate_to: str | None = None,
    ) -> None:
        super().__init__()
        self._clips = [c.clone() for c in clips]
        self._output = output
        self._language = language
        self._translate_to = translate_to
        self._cancelled = False

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            if not self._clips:
                self.failed.emit("Add a video clip to the timeline first.")
                return

            self.status.emit("Extracting audio…")
            self.progress.emit(5)
            tmp_dir = tempfile.TemporaryDirectory(prefix="cove-asr-")
            wav_path = Path(tmp_dir.name) / "audio.wav"
            extract_timeline_wav(self._clips, wav_path)
            if self._cancelled:
                return

            self.status.emit(
                "Transcribing… (first run downloads the speech model)"
            )
            self.progress.emit(20)
            cues = transcribe_wav(wav_path, self._language)
            if self._cancelled:
                return
            if not cues:
                self.failed.emit(
                    "No speech detected. Try another language, or check that "
                    "the clip has clear spoken audio."
                )
                return

            if self._translate_to and self._translate_to != self._language:
                if self._cancelled:
                    return
                self.status.emit(f"Translating to {self._translate_to}…")
                self.progress.emit(70)
                cues = translate_cues(
                    cues, self._language, self._translate_to,
                    status_cb=self.status.emit,
                    progress_cb=lambda done, total: self.progress.emit(
                        70 + int(20 * done / total) if total else 90
                    ),
                )

            self.progress.emit(90)
            self.status.emit("Writing SRT…")
            self._output.parent.mkdir(parents=True, exist_ok=True)
            self._output.write_text(cues_to_srt(cues), encoding="utf-8")
            self.progress.emit(100)
            self.status.emit(f"Saved {self._output.name}")
            self.finished.emit(self._output)
        except ff.FFmpegMissingError as exc:
            self.failed.emit(str(exc))
        except subprocess.CalledProcessError as exc:
            self.failed.emit(f"ffmpeg failed while extracting audio: {exc}")
        except Exception as exc:  # noqa: BLE001
            if self._cancelled:
                return
            self.failed.emit(str(exc))
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()


def start_transcribe(
    clips: list[Clip],
    output: Path,
    language: str,
    translate_to: str | None = None,
) -> tuple[QThread, TranscribeWorker]:
    thread = QThread()
    worker = TranscribeWorker(clips, output, language, translate_to)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    return thread, worker
