"""Karaoke ASS caption engine for viral Shorts, chapter lower-thirds for Documentaries, and SRT writer."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

from .audio_engine import BeatAudio, WordTiming
from .config import DOC_HEIGHT, DOC_WIDTH, SHORTS_HEIGHT, SHORTS_WIDTH

# Candidate bold font files across Linux (Debian/Ubuntu/GitHub Actions), macOS, and Windows
BOLD_FONT_PATHS = (
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Black.ttf",
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Black.ttf",
    "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)

PREFERRED_ASS_FAMILIES = ("Roboto", "DejaVu Sans", "Liberation Sans", "Noto Sans", "Arial")


def detect_bold_font_path() -> Path | None:
    """Return the sharpest available bold/black TrueType font file on this OS."""
    for candidate in BOLD_FONT_PATHS:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def detect_ass_font_family() -> str:
    """Detect an installed bold sans-serif font family name for libass."""
    try:
        proc = subprocess.run(
            ["fc-list", ":", "family"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            installed = {line.strip().split(",")[0].casefold(): line.strip().split(",")[0] for line in proc.stdout.splitlines() if line.strip()}
            for pref in PREFERRED_ASS_FAMILIES:
                if pref.casefold() in installed:
                    return installed[pref.casefold()]
    except (OSError, subprocess.SubprocessError):
        pass
    path = detect_bold_font_path()
    if path:
        if "roboto" in path.name.lower():
            return "Roboto"
        if "dejavu" in path.name.lower():
            return "DejaVu Sans"
        if "liberation" in path.name.lower():
            return "Liberation Sans"
    return "DejaVu Sans"


def escape_ass_text(text: str) -> str:
    """Escape special ASS override tokens while preserving clean readable punctuation."""
    cleaned = " ".join(str(text or "").split())
    return cleaned.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def format_ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc (centiseconds)."""
    total_cs = max(0, int(round(float(seconds) * 100.0)))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def group_words_into_bursts(words: list[WordTiming], *, max_words: int = 3, max_chars: int = 19) -> list[list[WordTiming]]:
    """Chunk word timings into crisp 2-3 word bursts optimized for fast vertical-video reading."""
    bursts: list[list[WordTiming]] = []
    current: list[WordTiming] = []
    current_len = 0

    for item in words:
        token = item.word.strip()
        if not token:
            continue
        projected = current_len + len(token) + (1 if current else 0)
        if current and (len(current) >= max_words or projected > max_chars):
            bursts.append(current)
            current = []
            current_len = 0

        current.append(item)
        current_len += len(token) + (1 if len(current) > 1 else 0)

        # Break cleanly right after major sentence/clause punctuation
        if token[-1] in ".!?;:" or (token[-1] == "," and len(current) >= 2):
            bursts.append(current)
            current = []
            current_len = 0

    if current:
        # Avoid orphan single short word at the end if previous burst has room
        if len(current) == 1 and bursts and len(bursts[-1]) < max_words + 1 and len(current[0].word) <= 6:
            bursts[-1].extend(current)
        else:
            bursts.append(current)
    return bursts


def generate_shorts_karaoke_ass(
    beats: list[BeatAudio],
    out_ass: Path,
    *,
    pill_label: str = "",
    hook_banner: str = "",
    width: int = SHORTS_WIDTH,
    height: int = SHORTS_HEIGHT,
) -> Path:
    """Generate dynamic word-by-word gold-highlighted ASS subtitles + 0.0-1.85s scroll-stopper hook banner."""
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    font_family = detect_ass_font_family()

    # ASS colors use &HAABBGGRR format:
    # Pure white: &H00FFFFFF, Warm viral gold (#FFD43B): &H003BD4FF, Charcoal outline: &H000A0A0A
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font_family},84,&H00FFFFFF,&H003BD4FF,&H000A0A0A,&H96000000,-1,0,0,0,100,100,1,0,1,6,3,2,64,64,475,1
Style: TopPill,{font_family},36,&H003BD4FF,&H00FFFFFF,&H55101418,&H00000000,-1,0,0,0,100,100,2,0,3,16,0,8,60,60,145,1
Style: HookBanner,{font_family},108,&H003BD4FF,&H00FFFFFF,&H00080A0F,&HA0000000,-1,0,0,0,100,100,2,0,1,9,4,8,56,56,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    if pill_label.strip():
        clean_pill = escape_ass_text(pill_label.strip().upper()[:30])
        events.append(
            f"Dialogue: 1,{format_ass_time(0.0)},{format_ass_time(2.60)},TopPill,,0,0,0,,"
            f"{{\\fad(120,260)}}●  {clean_pill}"
        )

    if hook_banner.strip():
        # Instant 0.0-1.85s visual scroll-stopper with quick pop-in scale animation (grabs muted scrollers in 0.3s!)
        clean_hook = escape_ass_text(hook_banner.strip().upper()[:24])
        events.append(
            f"Dialogue: 2,{format_ass_time(0.0)},{format_ass_time(1.85)},HookBanner,,0,0,0,,"
            f"{{\\fscx88\\fscy88\\t(0,110,\\fscx104\\fscy104)\\t(110,180,\\fscx100\\fscy100)\\fad(0,180)}}{clean_hook}"
        )

    all_bursts: list[list[WordTiming]] = []
    for beat in beats:
        if beat.words:
            all_bursts.extend(group_words_into_bursts(beat.words, max_words=3, max_chars=19))

    for b_idx, burst in enumerate(all_bursts):
        next_burst_start = all_bursts[b_idx + 1][0].start if b_idx + 1 < len(all_bursts) else None
        for w_idx, word_obj in enumerate(burst):
            ev_start = max(0.0, word_obj.start)
            if w_idx + 1 < len(burst):
                ev_end = max(ev_start + 0.05, burst[w_idx + 1].start)
            else:
                ev_end = max(ev_start + 0.08, word_obj.end + 0.04)
                if next_burst_start is not None:
                    ev_end = min(ev_end, max(ev_start + 0.05, next_burst_start - 0.01))

            styled_tokens: list[str] = []
            for j, token_obj in enumerate(burst):
                txt = escape_ass_text(token_obj.word.upper())
                if j == w_idx:
                    # Highlight actively spoken word in vibrant gold with crisp micro-pop
                    styled_tokens.append(f"{{\\c&H003BD4FF&\\fscx106\\fscy106}}{txt}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}")
                else:
                    styled_tokens.append(txt)

            line_text = " ".join(styled_tokens)
            events.append(
                f"Dialogue: 0,{format_ass_time(ev_start)},{format_ass_time(ev_end)},Karaoke,,0,0,0,,{line_text}"
            )

    out_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_ass


def generate_documentary_chapters_ass(
    chapters: list[tuple[float, float, int, str]],
    out_ass: Path,
    *,
    width: int = DOC_WIDTH,
    height: int = DOC_HEIGHT,
) -> Path:
    """Generate elegant cinematic lower-third chapter title cards for 16:9 Documentaries.

    Each tuple in `chapters` is (start_sec, duration_sec, chapter_number, chapter_title).
    """
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    font_family = detect_ass_font_family()
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ChapterKicker,{font_family},26,&H003BD4FF,&H00FFFFFF,&H00080A0C,&H80000000,-1,0,0,0,100,100,4,0,1,3,1,1,96,96,156,1
Style: ChapterTitle,{font_family},54,&H00FFFFFF,&H003BD4FF,&H00080A0C,&H90000000,-1,0,0,0,100,100,1,0,1,4,2,1,96,96,92,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for start_sec, dur_sec, num, title in chapters:
        show_start = max(0.0, start_sec + 0.35)
        show_end = min(start_sec + max(2.5, dur_sec - 0.5), show_start + 4.4)
        if show_end <= show_start + 0.5:
            continue
        kicker = f"CHAPTER {num:02d}"
        clean_title = escape_ass_text(title.strip())
        events.append(
            f"Dialogue: 1,{format_ass_time(show_start)},{format_ass_time(show_end)},ChapterKicker,,0,0,0,,"
            f"{{\\fad(350,450)}}{kicker}"
        )
        events.append(
            f"Dialogue: 1,{format_ass_time(show_start)},{format_ass_time(show_end)},ChapterTitle,,0,0,0,,"
            f"{{\\fad(350,450)}}{clean_title}"
        )

    out_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_ass


def beats_to_srt_cues(beats: list[BeatAudio], *, max_words_per_cue: int = 12) -> list[tuple[float, float, str]]:
    """Build clean sentence/clause-level SRT cues from BeatAudio timelines for YouTube CC."""
    cues: list[tuple[float, float, str]] = []
    for beat in beats:
        if beat.words and len(beat.words) > max_words_per_cue:
            # Split long documentary chapter narrations cleanly into natural subtitle cues
            chunks = group_words_into_bursts(beat.words, max_words=max_words_per_cue, max_chars=74)
            for chunk in chunks:
                text = " ".join(w.word for w in chunk).strip()
                if text:
                    cues.append((chunk[0].start, max(chunk[0].start + 0.4, chunk[-1].end), text))
        else:
            text = " ".join(re.split(r"\s+", beat.text)).strip()
            if text:
                cues.append((beat.start, max(beat.start + 0.4, beat.start + beat.duration), text))
    return cues


def write_srt(cues: Iterable[tuple[float, float, str]], out_srt: Path) -> Path:
    """Write standard UTF-8 SubRip (.srt) caption file."""
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for idx, (start, end, text) in enumerate(cues, start=1):
        clean = " ".join(str(text or "").split()).strip()
        if not clean:
            continue
        blocks.append(f"{idx}\n{format_srt_time(start)} --> {format_srt_time(max(start + 0.25, end))}\n{clean}")
    out_srt.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return out_srt
