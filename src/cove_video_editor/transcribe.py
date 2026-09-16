"""Generate SRT subtitles from timeline video audio via faster-whisper
(FunASR for Chinese, which transcribes and punctuates Mandarin far more
reliably than Whisper)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
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


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from a project-root ``.env`` file into
    ``os.environ``, without overwriting a variable that's already set (e.g.
    via ``setx``). Lets a secret like ``GEMINI_API_KEY`` live in a local,
    gitignored file instead of a real environment variable or source code.
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
        return


_load_dotenv()


def _ensure_model_cache_env() -> None:
    """Point faster-whisper/FunASR's downloaders at ``assets/models`` next
    to the project checkout, matching ``__main__.py``'s ``_model_cache_dir``
    — but via ``setdefault``, so it's a no-op whenever ``__main__.py`` (the
    GUI app) already set these earlier at startup.

    This only matters for code that imports this module *without* going
    through ``__main__.py`` first — a standalone dev script (``scripts/*``).
    Without it, ``huggingface_hub``/``modelscope`` fall back to caching in
    the user's home directory (``~/.cache``, almost always the C: drive on
    Windows regardless of where the project lives), and a script run this
    way ends up downloading its own separate multi-GB copy of a model the
    GUI app already cached under the project.
    """
    base = Path(__file__).resolve().parent.parent.parent / "assets" / "models"
    os.environ.setdefault("HF_HOME", str(base / "huggingface"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(base / "modelscope"))


_ensure_model_cache_env()

# Format combo label → Whisper language code.
SUBTITLE_LANG_FORMATS: dict[str, str] = {
    "English (.srt)": "en",
    "中文 Chinese (.srt)": "zh",
    "Tiếng Việt (.srt)": "vi",
}

# "Translate to" combo label → language code. Translation goes through the
# Gemini API (requires GEMINI_API_KEY) — see translate_cues().
# ``None`` means keep the transcript in the spoken (source) language.
TRANSLATE_LANG_FORMATS: dict[str, str | None] = {
    "None (keep original)": None,
    "English": "en",
    "中文 Chinese": "zh",
    "Tiếng Việt": "vi",
}

_MODEL_NAME = "base"

# FunASR (Paraformer) sentence-splitting tuning — see transcribe_wav_funasr().
# Every clause-ending mark (sentence enders *and* commas) ends a cue, so
# each spoken clause becomes its own subtitle card instead of several
# clauses — each of which may become its own sentence once translated —
# being crammed into one long-lived cue.
_CJK_RE = re.compile(r"[一-鿿]")
_LATIN_RUN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_END = set("。！？…，、；")

# faster-whisper word-splitting tuning — see _whisper_words_to_cues(). Whisper's
# own segments span whatever the VAD chunked together, which is often several
# spoken clauses *with a silent pause between them* crammed into one cue with
# a start/end that stretches across the pause. Rebuilding cues from
# word-level timestamps instead — cutting a new cue at a clause-ending mark
# *or* whenever the gap to the next word is long enough to be an actual
# pause — keeps each cue to one continuous run of speech and its timing
# tight around it, mirroring what _funasr_text_to_cues() already does for
# Chinese.
_LATIN_CLAUSE_END = set(".!?,;:…")
_WHISPER_PAUSE_GAP_S = 0.5
_WHISPER_MAX_CUE_DURATION_S = 7.0


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


def _funasr_text_to_cues(
    text: str, timestamps_ms: list[list[int]]
) -> list[tuple[float, float, str]]:
    """Split FunASR's globally-punctuated ``text`` into sentence cues.

    FunASR's own per-chunk ``sentence_info`` splits at VAD segment
    boundaries, which shifts a boundary character into the wrong chunk
    whenever the recognizer's context window crosses a chunk edge.
    Splitting the single global ``text`` string against the flat
    per-token ``timestamp`` list avoids that: each CJK character consumes
    one timestamp entry, and each run of consecutive ASCII letters/digits
    (e.g. "Oppo", "max") consumes exactly one, since that's how the model
    tokenized them. Every comma/clause-end mark closes a cue, so each
    spoken clause is its own cue — important once translated, since one
    Chinese clause commonly becomes one full sentence in the target
    language.
    """
    cues: list[tuple[float, float, str]] = []
    ts_index = 0
    buf: list[str] = []
    buf_start_ms: int | None = None
    buf_end_ms: int | None = None

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        m = _LATIN_RUN_RE.match(text, i)
        if m:
            token = m.group()
            i = m.end()
        else:
            token = ch
            i += 1

        if _CJK_RE.fullmatch(token) or _LATIN_RUN_RE.fullmatch(token):
            if ts_index >= len(timestamps_ms):
                start_ms, end_ms = buf_end_ms or 0, buf_end_ms or 0
            else:
                start_ms, end_ms = timestamps_ms[ts_index]
                ts_index += 1
            if buf_start_ms is None:
                buf_start_ms = start_ms
            buf_end_ms = end_ms
            buf.append(token)
        else:
            buf.append(token)
            if token in _SENTENCE_END:
                sentence = "".join(buf).strip()
                if sentence and buf_start_ms is not None:
                    cues.append(
                        (buf_start_ms / 1000.0, buf_end_ms / 1000.0, sentence)
                    )
                buf = []
                buf_start_ms = None
                buf_end_ms = None

    tail = "".join(buf).strip()
    if tail and buf_start_ms is not None:
        cues.append((buf_start_ms / 1000.0, buf_end_ms / 1000.0, tail))
    return cues


_funasr_model = None


def _get_funasr_model():
    global _funasr_model
    if _funasr_model is None:
        try:
            from funasr import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "funasr is not installed. Run:\n"
                "  pip install funasr kaldi-native-fbank"
            ) from exc
        _funasr_model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            device="cpu",
            disable_update=True,
        )
    return _funasr_model


def transcribe_wav_funasr(wav_path: Path) -> list[tuple[float, float, str]]:
    """Run FunASR (Paraformer + VAD + punctuation) for Mandarin Chinese.

    Whisper frequently garbles Mandarin and never restores punctuation,
    since Chinese text has no spaces to hint at word/sentence boundaries.
    FunASR's Paraformer + ct-punc pipeline is purpose-built for Mandarin
    and produces properly punctuated, far more accurate transcripts.
    """
    model = _get_funasr_model()
    result = model.generate(input=str(wav_path), batch_size_s=300)

    cues: list[tuple[float, float, str]] = []
    for item in result:
        text = (item.get("text") or "").strip()
        timestamps_ms = item.get("timestamp") or []
        if not text:
            continue
        cues.extend(_funasr_text_to_cues(text, timestamps_ms))
    return cues


def _whisper_words_to_cues(
    segments,  # noqa: ANN001 — Iterable[faster_whisper.transcribe.Segment]
    pause_gap_s: float = _WHISPER_PAUSE_GAP_S,
    max_cue_duration_s: float = _WHISPER_MAX_CUE_DURATION_S,
) -> list[tuple[float, float, str]]:
    """Rebuild cues from Whisper's per-word timestamps.

    A new cue starts whenever the silent gap since the previous word is at
    least ``pause_gap_s`` (a real spoken pause, not just VAD noise), and the
    current cue is closed as soon as a word ends in clause-ending
    punctuation or the cue has been running for ``max_cue_duration_s`` —
    whichever comes first — so a long unpunctuated run of speech still gets
    split into shorter cues.
    """
    cues: list[tuple[float, float, str]] = []
    buf: list[str] = []
    buf_start: float | None = None
    buf_end: float | None = None

    def flush() -> None:
        nonlocal buf, buf_start, buf_end
        text = "".join(buf).strip()
        if text and buf_start is not None and buf_end is not None:
            cues.append((buf_start, buf_end, text))
        buf = []
        buf_start = None
        buf_end = None

    for seg in segments:
        for w in seg.words or []:
            start, end = float(w.start), float(w.end)
            if buf_end is not None and (start - buf_end) >= pause_gap_s:
                flush()
            if buf_start is None:
                buf_start = start
            buf_end = end
            buf.append(w.word)
            stripped = w.word.strip()
            ends_clause = bool(stripped) and stripped[-1] in _LATIN_CLAUSE_END
            if ends_clause or (buf_end - buf_start) >= max_cue_duration_s:
                flush()
    flush()
    return cues


def transcribe_wav(wav_path: Path, language: str) -> list[tuple[float, float, str]]:
    """Return ``(start, end, text)`` cues for ``wav_path``.

    Chinese goes through FunASR (see ``transcribe_wav_funasr``); every
    other language goes through faster-whisper.
    """
    if language == "zh":
        return transcribe_wav_funasr(wav_path)

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
        word_timestamps=True,
    )
    return _whisper_words_to_cues(segments)


_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
_GEMINI_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "vi": "Vietnamese",
}

# Cues per Gemini request. Kept well under what the model's context window
# could take: past a few hundred numbered lines in one prompt, Gemini gets
# noticeably more likely to merge/drop a line, throwing off the 1:1 mapping
# back to cue timings (checked below via the returned-count assertion). Most
# single-clip transcripts run under this, so they cost exactly one request;
# only unusually long ones split into more.
_GEMINI_BATCH_SIZE = 200
# HTTP codes worth retrying: rate-limited or a transient server-side hiccup,
# as opposed to e.g. 400 (bad request) or 403 (bad API key), which won't
# succeed no matter how many times they're retried.
_GEMINI_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class _GeminiTransientError(RuntimeError):
    """A Gemini call failed in a way that's worth retrying (rate limit or
    transient server error), as opposed to a permanent failure like a bad
    API key or malformed request.
    """


def _gemini_translate_batch(
    texts: list[str], from_lang: str, target_lang: str, api_key: str
) -> list[str]:
    from_name = _GEMINI_LANG_NAMES.get(from_lang, from_lang)
    to_name = _GEMINI_LANG_NAMES.get(target_lang, target_lang)
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"You are translating video subtitles from {from_name} to {to_name}. "
        f"Translate each numbered line below into natural, colloquial "
        f"{to_name} suitable for on-screen subtitles, preserving the tone, "
        f"slang, and meaning of casual spoken {from_name} rather than "
        "translating word-for-word. Return exactly "
        f"{len(texts)} translations, in the same order, as a JSON array of "
        "strings — one string per input line, with no numbering and no "
        "extra commentary.\n\n" + numbered
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
    }).encode("utf-8")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_GEMINI_MODEL}:generateContent"
    )
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error_cls = (
            _GeminiTransientError if exc.code in _GEMINI_RETRYABLE_CODES
            else RuntimeError
        )
        raise error_cls(f"Gemini API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise _GeminiTransientError(
            f"Could not reach Gemini API: {exc.reason}"
        ) from exc

    try:
        text_out = payload["candidates"][0]["content"]["parts"][0]["text"]
        translations = json.loads(text_out)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {payload}") from exc
    if len(translations) != len(texts):
        raise RuntimeError(
            f"Gemini returned {len(translations)} line(s), expected {len(texts)}."
        )
    return translations


def _translate_cues_gemini(
    cues: list[tuple[float, float, str]],
    from_lang: str,
    target_lang: str,
    api_key: str,
    status_cb=None,  # noqa: ANN001
    progress_cb=None,  # noqa: ANN001
) -> list[tuple[float, float, str]]:
    """Translate cue text via the Gemini API — an actual LLM, so it handles
    casual/slangy speech and specific terminology far better than a plain
    NMT model. Requires ``api_key`` and an internet connection; cues are
    sent in batches so the model sees enough surrounding context to
    translate consistently.

    A batch that fails with a rate limit or a transient server error (5xx)
    is retried with backoff — Gemini's API is flaky enough under load that
    a single hiccup on, say, batch 2 of 3 would otherwise abort the whole
    transcript partway through.
    """
    import time

    total = len(cues)
    out: list[tuple[float, float, str]] = []
    for start_idx in range(0, total, _GEMINI_BATCH_SIZE):
        batch = cues[start_idx : start_idx + _GEMINI_BATCH_SIZE]
        if status_cb:
            status_cb(
                f"Translating via Gemini… ({start_idx + len(batch)}/{total})"
            )
        texts = [c[2] for c in batch]
        delay = 2.0
        for attempt in range(5):
            try:
                translations = _gemini_translate_batch(
                    texts, from_lang, target_lang, api_key
                )
                break
            except _GeminiTransientError:
                if attempt == 4:
                    raise
                if status_cb:
                    status_cb(
                        "Gemini is busy/rate-limited, retrying "
                        f"({attempt + 1}/5)…"
                    )
                time.sleep(delay)
                delay *= 2
        for (start, end, original), new_text in zip(batch, translations):
            new_text = (new_text or "").strip()
            out.append((start, end, new_text or original))
        if progress_cb:
            progress_cb(min(start_idx + _GEMINI_BATCH_SIZE, total), total)
    return out


def translate_cues(
    cues: list[tuple[float, float, str]],
    from_lang: str,
    target_lang: str,
    status_cb=None,  # noqa: ANN001
    progress_cb=None,  # noqa: ANN001
) -> list[tuple[float, float, str]]:
    """Translate cue text from ``from_lang`` to ``target_lang`` via the
    Gemini API, keeping original timing. Requires a ``GEMINI_API_KEY``
    environment variable (or ``.env`` entry) and an internet connection.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the "
            "project root, or set it as an environment variable."
        )
    return _translate_cues_gemini(
        cues, from_lang, target_lang, api_key,
        status_cb=status_cb, progress_cb=progress_cb,
    )


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

            if self._language == "zh":
                self.status.emit(
                    "Transcribing… (first run downloads the FunASR "
                    "Chinese speech + punctuation models, ~2 GB)"
                )
            else:
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
