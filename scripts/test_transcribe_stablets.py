"""Standalone smoke test: video -> .srt using stable-ts (regrouped word-level
stabilization on top of faster-whisper large-v3).

Usage:
    python scripts/test_transcribe_stablets.py <video_path> [--lang zh] [--out out.srt]
        [--model large-v3]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:  # Windows consoles default to cp1252, which can't print zh/vi text.
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from cove_video_editor import ffmpeg_utils as ff  # noqa: E402


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
    parser.add_argument("--lang", default="zh", help="Spoken language code (default: zh)")
    parser.add_argument("--model", default="large-v3", help="faster-whisper model size (default: large-v3)")
    parser.add_argument("--out", type=Path, default=None, help="Output .srt path")
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.out or video_path.with_suffix(f".stablets.{args.lang}.srt")

    import stable_whisper

    print(f"[1/3] Loading {args.model} (faster-whisper backend, cache under $HF_HOME)...")
    model = stable_whisper.load_faster_whisper(
        args.model, device="cpu", compute_type="int8"
    )

    with tempfile.TemporaryDirectory(prefix="cove-stablets-") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        print("[2/3] Extracting audio...")
        extract_wav(video_path, wav_path)

        print(f"[3/3] Transcribing ({args.lang}) with stable-ts regrouping...")
        result = model.transcribe(
            str(wav_path),
            language=args.lang,
            vad=True,
            verbose=False,
        )

    result.to_srt_vtt(str(out_path), word_level=False)
    print(f"Wrote {out_path} ({len(result.segments)} cue(s))")


if __name__ == "__main__":
    main()
