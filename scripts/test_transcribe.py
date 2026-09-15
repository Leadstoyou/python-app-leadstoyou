"""Standalone smoke test: video path -> .srt, without the Qt timeline.

Usage:
    python scripts/test_transcribe.py <video_path> [--lang vi|en|zh] [--out out.srt]
        [--translate-to vi|en|zh]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cove_video_editor import ffmpeg_utils as ff  # noqa: E402
from cove_video_editor.transcribe import (  # noqa: E402
    cues_to_srt,
    transcribe_wav,
    translate_cues,
)


def extract_wav(video_path: Path, wav_path: Path) -> None:
    ffmpeg = ff.require_ffmpeg()
    cmd = [
        ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument("--lang", default="vi", help="Whisper language code (default: vi)")
    parser.add_argument("--out", type=Path, default=None, help="Output .srt path")
    parser.add_argument(
        "--translate-to",
        default=None,
        help="Also write a second .srt translated to this language code "
        "(e.g. vi, en, zh) via Gemini (requires GEMINI_API_KEY).",
    )
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.out or video_path.with_suffix(".srt")

    with tempfile.TemporaryDirectory(prefix="cove-asr-test-") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        print(f"[1/3] Extracting audio -> {wav_path}")
        extract_wav(video_path, wav_path)

        print("[2/3] Transcribing (first run downloads the speech model)...")
        cues = transcribe_wav(wav_path, args.lang)
        print(f"      {len(cues)} cue(s) found")

        if not cues:
            raise SystemExit("No speech detected.")

        print(f"[3/3] Writing SRT -> {out_path}")
        out_path.write_text(cues_to_srt(cues), encoding="utf-8")

    if args.translate_to and args.translate_to != args.lang:
        translated_out = out_path.with_suffix(f".{args.translate_to}.srt")
        print(f"[4/4] Translating {args.lang} -> {args.translate_to}...")
        translated_cues = translate_cues(
            cues, args.lang, args.translate_to, status_cb=print
        )
        translated_out.write_text(cues_to_srt(translated_cues), encoding="utf-8")
        print(f"      Wrote {translated_out}")

    print("Done.")


if __name__ == "__main__":
    main()
