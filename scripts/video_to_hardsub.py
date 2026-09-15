"""End-to-end smoke test: video -> transcribe -> translate -> burn subtitles
into a new video file (hardsub), no Qt timeline involved.

Usage:
    python scripts/video_to_hardsub.py <video_path> --from-lang zh --to-lang vi [--out out.mp4]
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
from cove_video_editor.clip import SubtitleTrack  # noqa: E402
from cove_video_editor.exporter import _render_ass  # noqa: E402
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


def burn_subtitles(
    video_path: Path, ass_path: Path, out_path: Path, has_audio: bool
) -> None:
    ffmpeg = ff.require_ffmpeg()
    sub_arg = ff.escape_filter_arg(str(ass_path))
    cmd = [
        ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"subtitles='{sub_arg}'",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
    ]
    cmd += ["-c:a", "copy"] if has_audio else ["-an"]
    cmd += [str(out_path)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument("--from-lang", default="zh", help="Spoken language code (default: zh)")
    parser.add_argument("--to-lang", default="vi", help="Subtitle target language code (default: vi)")
    parser.add_argument("--out", type=Path, default=None, help="Output video path")
    parser.add_argument("--font-size", type=int, default=36, help="Burn-in font size in output pixels (default: 36)")
    parser.add_argument("--keep-srt", action="store_true", help="Also save the translated .srt next to the output video")
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    out_path = args.out or video_path.with_name(
        f"{video_path.stem}.hardsub.{args.to_lang}{video_path.suffix}"
    )

    info = ff.probe(video_path)
    print(f"[1/5] Probed video: {info.width}x{info.height}, {info.duration:.1f}s")

    with tempfile.TemporaryDirectory(prefix="cove-hardsub-") as tmp:
        tmp_dir = Path(tmp)
        wav_path = tmp_dir / "audio.wav"
        print("[2/5] Extracting audio...")
        extract_wav(video_path, wav_path)

        print(f"[3/5] Transcribing ({args.from_lang})...")
        cues = transcribe_wav(wav_path, args.from_lang)
        print(f"      {len(cues)} cue(s) found")
        if not cues:
            raise SystemExit("No speech detected.")

        if args.to_lang and args.to_lang != args.from_lang:
            print(f"[4/5] Translating {args.from_lang} -> {args.to_lang} via Gemini...")
            cues = translate_cues(cues, args.from_lang, args.to_lang, status_cb=print)
        else:
            print("[4/5] Skipping translation (same language).")

        if args.keep_srt:
            srt_path = out_path.with_suffix(f".{args.to_lang}.srt")
            srt_path.write_text(cues_to_srt(cues), encoding="utf-8")
            print(f"      Wrote {srt_path}")

        sub = SubtitleTrack(
            path=video_path, font_size=args.font_size, cues=cues, active=True,
        )
        ass_path = tmp_dir / "burn.ass"
        ass_path.write_text(_render_ass(sub, info.width, info.height), encoding="utf-8")

        print(f"[5/5] Burning subtitles -> {out_path}")
        burn_subtitles(video_path, ass_path, out_path, info.has_audio)

    print("Done.")


if __name__ == "__main__":
    main()
