"""Standalone smoke test: translate an existing .srt file's cues.

Usage:
    python scripts/test_translate.py <srt_path> --from-lang zh --to-lang vi [--out out.srt]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:  # Windows consoles default to cp1252, which can't print zh/vi text.
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from cove_video_editor.transcribe import cues_to_srt, translate_cues  # noqa: E402

_TS_RE = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)")


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        ts_line_idx = next((i for i, l in enumerate(lines) if _TS_RE.search(l)), None)
        if ts_line_idx is None:
            continue
        m = _TS_RE.search(lines[ts_line_idx])
        assert m is not None
        start = _ts_to_seconds(*m.groups()[0:4])
        end = _ts_to_seconds(*m.groups()[4:8])
        body = "\n".join(lines[ts_line_idx + 1 :]).strip()
        if body:
            cues.append((start, end, body))
    return cues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("srt", type=Path, help="Path to input .srt file")
    parser.add_argument("--from-lang", required=True, help="Source language code (e.g. zh)")
    parser.add_argument("--to-lang", required=True, help="Target language code (e.g. vi)")
    parser.add_argument("--out", type=Path, default=None, help="Output .srt path")
    args = parser.parse_args()

    srt_path: Path = args.srt
    if not srt_path.is_file():
        raise SystemExit(f"SRT not found: {srt_path}")

    out_path = args.out or srt_path.with_suffix(f".{args.to_lang}.srt")

    cues = parse_srt(srt_path.read_text(encoding="utf-8"))
    print(f"[1/2] Parsed {len(cues)} cue(s) from {srt_path}")
    if not cues:
        raise SystemExit("No cues found in the input file.")

    print(f"[2/2] Translating {args.from_lang} -> {args.to_lang}...")
    translated = translate_cues(cues, args.from_lang, args.to_lang, status_cb=print)

    out_path.write_text(cues_to_srt(translated), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
