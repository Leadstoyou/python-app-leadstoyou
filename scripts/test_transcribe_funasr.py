"""Standalone smoke test: video -> .srt using FunASR (Paraformer-zh + VAD + punctuation).

Usage:
    python scripts/test_transcribe_funasr.py <video_path> [--out out.srt]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cove_video_editor import ffmpeg_utils as ff  # noqa: E402
from cove_video_editor.transcribe import cues_to_srt  # noqa: E402


def extract_wav(video_path: Path, wav_path: Path) -> None:
    ffmpeg = ff.require_ffmpeg()
    cmd = [
        ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


_CJK_RE = re.compile(r"[一-鿿]")
_LATIN_RUN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_END = set("。！？…，、；")


def _text_to_cues(
    text: str, timestamps_ms: list[list[int]]
) -> list[tuple[float, float, str]]:
    """Split FunASR's globally-punctuated ``text`` into sentence cues.

    ``sentence_info`` splits per VAD chunk, which shifts a boundary
    character into the wrong chunk when the recognizer's context window
    crosses a chunk edge. Splitting the single global ``text`` string
    against the flat per-token ``timestamp`` list avoids that: each CJK
    character consumes one timestamp entry, and each run of consecutive
    ASCII letters/digits (e.g. "Oppo", "max") consumes exactly one, since
    that's how the model tokenized them.
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


def transcribe_funasr(wav_path: Path) -> list[tuple[float, float, str]]:
    from funasr import AutoModel

    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True,
    )
    result = model.generate(input=str(wav_path), batch_size_s=300)

    cues: list[tuple[float, float, str]] = []
    for item in result:
        text = (item.get("text") or "").strip()
        timestamps_ms = item.get("timestamp") or []
        if not text:
            continue
        cues.extend(_text_to_cues(text, timestamps_ms))
    return cues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument("--out", type=Path, default=None, help="Output .srt path")
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.out or video_path.with_suffix(".funasr.srt")

    with tempfile.TemporaryDirectory(prefix="cove-asr-funasr-") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        print(f"[1/3] Extracting audio -> {wav_path}")
        extract_wav(video_path, wav_path)

        print("[2/3] Transcribing with FunASR (first run downloads models)...")
        cues = transcribe_funasr(wav_path)
        print(f"      {len(cues)} cue(s) found")

        if not cues:
            raise SystemExit("No speech detected.")

        print(f"[3/3] Writing SRT -> {out_path}")
        out_path.write_text(cues_to_srt(cues), encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    main()
