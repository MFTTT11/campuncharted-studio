"""Pure-FFmpeg low-RAM cinematic renderer for 9:16 Shorts and 16:9 Documentaries."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .config import DOC_HEIGHT, DOC_WIDTH, SHORTS_HEIGHT, SHORTS_WIDTH, VIDEO_FPS
from .media_engine import MediaAsset


@dataclass(frozen=True)
class VisualCut:
    index: int
    asset: MediaAsset
    source_start: float
    duration: float
    motion_mode: int
    punch_zoom: bool = False


def _escape_ffmpeg_filter_path(path: Path) -> str:
    raw = str(path.resolve()).replace("\\", "/")
    return raw.replace(":", r"\:").replace("'", r"\'")


def create_gradient_scrim(
    out_png: Path,
    *,
    width: int,
    height: int,
    top_ratio: float = 0.14,
    top_alpha: int = 105,
    bottom_ratio: float = 0.38,
    bottom_alpha: int = 160,
) -> Path:
    """Create a smooth transparent PNG readability gradient for subtitles and pills."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Build a 1px-wide vertical alpha strip and resize cleanly for speed
    strip = Image.new("RGBA", (1, height), (0, 0, 0, 0))
    px = strip.load()
    top_px = int(height * top_ratio)
    bot_start = int(height * (1.0 - bottom_ratio))

    for y in range(height):
        alpha = 0
        if top_px > 0 and y < top_px:
            frac = 1.0 - (y / float(top_px))
            alpha = max(alpha, int(top_alpha * (frac**1.6)))
        if y >= bot_start and height > bot_start:
            frac = (y - bot_start) / float(height - bot_start)
            alpha = max(alpha, int(bottom_alpha * (frac**1.45)))
        px[0, y] = (6, 9, 14, min(255, max(0, alpha)))

    full = strip.resize((width, height), Image.Resampling.NEAREST)
    full.save(out_png, "PNG")
    return out_png


def _probe_frame_metrics(
    mp4_path: Path,
    timestamp: float,
    out_png: Path,
    *,
    resize: tuple[int, int] | None = None,
) -> tuple[float, float, float] | None:
    """Extract one frame via FFmpeg and return (mean_luminance, stddev_contrast, mean_saturation)."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, timestamp):.3f}",
                "-i",
                str(mp4_path),
                "-frames:v",
                "1",
                str(out_png),
            ],
            check=True,
            timeout=25,
        )
        if not out_png.is_file():
            return None
        with Image.open(out_png) as raw:
            rgb = raw.convert("RGB")
            if resize:
                rgb = rgb.resize(resize, Image.Resampling.BILINEAR)
            hsv = rgb.convert("HSV")
            rgb_stat = ImageStat.Stat(rgb)
            hsv_stat = ImageStat.Stat(hsv)
            brightness = sum(rgb_stat.mean) / 3.0
            contrast = sum(rgb_stat.stddev) / 3.0
            saturation = hsv_stat.mean[1]
            return brightness, contrast, saturation
    except (OSError, subprocess.SubprocessError):
        return None


def _find_bright_start_offset(
    asset: MediaAsset,
    shot_dur: float,
    work_dir: Path | None,
    tag: str,
) -> float:
    """Pick the brightest non-fade-in start offset inside a stock clip so cuts open vividly."""
    max_start = max(0.0, asset.duration - shot_dur - 0.15)
    default_start = min(max_start, min(1.5, round(asset.duration * 0.10, 2)) if asset.duration >= 8.0 else 0.25)
    if work_dir is None or asset.local_path is None or not asset.local_path.is_file() or max_start < 0.8:
        return round(max(0.0, default_start), 3)

    probe_dir = work_dir / "cut_offsets"
    probe_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted({round(min(max_start, max(0.25, t)), 2) for t in (default_start, asset.duration * 0.28, asset.duration * 0.52)})
    best_ts = default_start
    best_score = float("-inf")
    for idx, ts in enumerate(candidates):
        metrics = _probe_frame_metrics(asset.local_path, ts, probe_dir / f"{tag}_{idx}.png", resize=(180, 180))
        if metrics is None:
            continue
        brightness, contrast, saturation = metrics
        dark_pen = (45.0 - brightness) * 2.2 if brightness < 45.0 else 0.0
        score = (contrast * 1.1) + (saturation * 0.5) + (brightness * 0.35) - dark_pen
        if score > best_score:
            best_score = score
            best_ts = ts
    return round(max(0.0, best_ts), 3)


def rank_clips_for_hook(
    clips: list[MediaAsset],
    work_dir: Path,
    *,
    min_brightness: float = 14.0,
    min_keep: int = 2,
    prefer_portrait: bool = True,
) -> list[MediaAsset]:
    """Order clips brightest/sharpest first and drop pitch-black stock clips when enough visible clips exist."""
    playable = [c for c in clips if c.local_path and c.local_path.is_file() and c.duration > 0.5]
    if len(playable) <= 1:
        return playable

    probe_dir = work_dir / "hook_rank"
    probe_dir.mkdir(parents=True, exist_ok=True)
    scored: list[tuple[float, bool, MediaAsset]] = []
    for idx, asset in enumerate(playable):
        assert asset.local_path is not None
        sample_times = sorted(
            {
                round(min(max(0.25, asset.duration * 0.25), max(0.25, asset.duration - 0.25)), 2),
                round(min(max(0.25, asset.duration * 0.60), max(0.25, asset.duration - 0.25)), 2),
            }
        )
        samples: list[tuple[float, float, float]] = []
        for s_idx, ts in enumerate(sample_times):
            metrics = _probe_frame_metrics(
                asset.local_path,
                ts,
                probe_dir / f"c_{idx:02d}_{s_idx}.png",
                resize=(240, 240),
            )
            if metrics is not None:
                samples.append(metrics)

        if samples:
            brightness = sum(m[0] for m in samples) / len(samples)
            worst_brightness = min(m[0] for m in samples)
            contrast = sum(m[1] for m in samples) / len(samples)
            saturation = sum(m[2] for m in samples) / len(samples)
            # Strongly penalize murky dark clips (<55 mean brightness) from opening a Short!
            dark_penalty = (55.0 - brightness) * 2.5 if brightness < 55.0 else 0.0
            worst_dark_penalty = (20.0 - worst_brightness) * 3.0 if worst_brightness < 20.0 else 0.0
            blowout_penalty = (brightness - 215.0) * 1.5 if brightness > 215.0 else 0.0
            orientation_bonus = 25.0 if (asset.is_portrait == prefer_portrait) else 0.0
            score = (
                (contrast * 1.2)
                + (saturation * 0.8)
                + orientation_bonus
                - dark_penalty
                - worst_dark_penalty
                - blowout_penalty
            )
            is_usable = worst_brightness >= min_brightness and contrast >= 6.0
        else:
            score = -1000.0
            is_usable = False
        scored.append((score, is_usable, asset))

    scored.sort(key=lambda item: item[0], reverse=True)
    usable = [asset for _, ok, asset in scored if ok]
    if len(usable) >= min_keep:
        dropped = len(scored) - len(usable)
        if dropped > 0:
            print(
                f"[VIDEO] Filtered {dropped} near-black/flat stock clip(s); keeping {len(usable)} visible clips",
                flush=True,
            )
        return usable
    return [asset for _, _, asset in scored]


def plan_beat_synced_cuts(
    beat_clips: list[list[MediaAsset]],
    beat_specs: list[tuple[float, float, bool]],
    total_duration: float,
    *,
    work_dir: Path | None = None,
    max_shot_seconds: float = 2.15,
) -> list[VisualCut]:
    """Plan visual cuts strictly synced to each spoken beat's exact narration timeline and matched footage.

    Each tuple in `beat_specs` is `(start_sec, duration_sec, punch_zoom)`.
    Long beats (>max_shot_seconds) and Beat 0 (when 2 hook clips exist) split into crisp rapid
    sub-cuts so the viewer never sees a static shot >2.15s while every frame matches the spoken beat.
    """
    flat_fallback = [
        c
        for bucket in beat_clips
        for c in bucket
        if c.local_path and c.local_path.is_file() and c.duration > 0.5
    ]
    if not flat_fallback:
        raise ValueError("No playable beat-synced local video clips available for rendering.")

    cuts: list[VisualCut] = []
    clip_cursors: dict[str, float] = {}
    cut_index = 0
    num_beats = len(beat_specs)

    for b_idx, (b_start, b_dur, b_punch) in enumerate(beat_specs):
        # Compute exact visual window for this beat up to the start of the next beat (or total_duration)
        if b_idx + 1 < num_beats:
            next_start = max(b_start + 0.6, beat_specs[b_idx + 1][0])
            window_dur = max(0.6, next_start - b_start)
        else:
            elapsed_so_far = sum(c.duration for c in cuts)
            window_dur = max(0.7, total_duration - elapsed_so_far)

        raw_bucket = beat_clips[b_idx] if b_idx < len(beat_clips) else []
        bucket = [c for c in raw_bucket if c.local_path and c.local_path.is_file() and c.duration > 0.5]
        if not bucket:
            bucket = [flat_fallback[b_idx % len(flat_fallback)]]

        # Split long beats (or opening hook with 2 clips) into brisk 1.2-2.1s rapid cuts
        if b_idx == 0 and len(bucket) >= 2 and window_dur >= 1.6:
            sub_count = 2
        elif window_dur > max_shot_seconds:
            sub_count = max(2, int(math.ceil(window_dur / max_shot_seconds)))
        else:
            sub_count = 1

        sub_dur = window_dur / sub_count
        for sub_idx in range(sub_count):
            asset = bucket[sub_idx % len(bucket)]
            dur = round(window_dur - sub_dur * (sub_count - 1), 3) if sub_idx == sub_count - 1 else round(sub_dur, 3)
            dur = max(0.55, dur)

            if asset.asset_key not in clip_cursors:
                cursor = _find_bright_start_offset(asset, dur, work_dir, f"b{b_idx:02d}_s{sub_idx}")
            else:
                cursor = clip_cursors[asset.asset_key]
            max_start = max(0.0, asset.duration - dur - 0.15)
            if cursor > max_start:
                cursor = (cursor * 0.43) % max(0.05, max_start) if max_start > 0.2 else 0.0
            clip_cursors[asset.asset_key] = cursor + dur + 1.25

            cuts.append(
                VisualCut(
                    index=cut_index,
                    asset=asset,
                    source_start=round(max(0.0, cursor), 3),
                    duration=dur,
                    motion_mode=cut_index % 4,
                    punch_zoom=bool((b_punch and sub_idx == 0) or cut_index == 0),
                )
            )
            cut_index += 1

    # Absorb tiny float rounding drift on final cut so stitched video always covers total_duration
    if cuts:
        drift = total_duration - sum(c.duration for c in cuts)
        if abs(drift) > 0.01:
            last = cuts[-1]
            cuts[-1] = VisualCut(
                index=last.index,
                asset=last.asset,
                source_start=last.source_start,
                duration=max(0.6, round(last.duration + drift, 3)),
                motion_mode=last.motion_mode,
                punch_zoom=last.punch_zoom,
            )
    return cuts


def plan_visual_cuts(
    clips: list[MediaAsset],
    total_duration: float,
    *,
    target_cut_seconds: float = 3.1,
    min_cut_seconds: float = 2.4,
    front_loaded_hook: bool = False,
) -> list[VisualCut]:
    """Build a dynamic cut schedule rotating across distinct clips with non-overlapping sub-ranges.

    When `front_loaded_hook=True` (Shorts), the opening ~6 seconds use rapid 1.7s snap-cuts
    to freeze mobile scrollers before settling into brisk ~2.4s escalation cuts.
    """
    playable = [c for c in clips if c.local_path and c.local_path.is_file() and c.duration > 0.5]
    if not playable:
        raise ValueError("No playable local video clips available for rendering.")

    durations: list[float] = []
    if front_loaded_hook and total_duration > 6.5:
        elapsed = 0.0
        # 3 rapid 1.75s scroll-stopping cuts across the opening ~5.25s hook window
        while elapsed < 5.2 and (total_duration - elapsed) > 3.2:
            durations.append(1.75)
            elapsed += 1.75
        tail = max(0.8, total_duration - elapsed)
        body_cuts = max(1, int(round(tail / target_cut_seconds)))
        body_dur = tail / body_cuts
        if body_dur < min_cut_seconds and body_cuts > 1:
            body_cuts = max(1, int(math.floor(tail / min_cut_seconds)))
            body_dur = tail / body_cuts
        for i in range(body_cuts):
            durations.append(round(body_dur, 3))
        # Absorb rounding drift on the final cut
        drift = total_duration - sum(durations)
        durations[-1] = max(0.8, round(durations[-1] + drift, 3))
    else:
        cut_count = max(1, int(round(total_duration / target_cut_seconds)))
        base_dur = total_duration / cut_count
        if base_dur < min_cut_seconds and cut_count > 1:
            cut_count = max(1, int(math.floor(total_duration / min_cut_seconds)))
            base_dur = total_duration / cut_count
        remaining = total_duration
        for idx in range(cut_count):
            dur = remaining if idx == cut_count - 1 else round(base_dur, 3)
            dur = max(0.8, dur)
            remaining = max(0.0, remaining - dur)
            durations.append(dur)

    # Skip common 0.5-1.5s fade-from-black intros on long stock clips while keeping short clips untouched
    clip_cursors: dict[str, float] = {
        c.asset_key: (min(1.5, round(c.duration * 0.10, 2)) if c.duration >= 8.0 else 0.25)
        for c in playable
    }
    cuts: list[VisualCut] = []

    for idx, dur in enumerate(durations):
        asset = playable[idx % len(playable)]
        cursor = clip_cursors[asset.asset_key]
        max_start = max(0.0, asset.duration - dur - 0.15)
        if cursor > max_start:
            # Wrap cleanly to a distinct offset if this clip is reused on a later pass
            cursor = (cursor * 0.37) % max(0.05, max_start) if max_start > 0.2 else 0.0
        clip_cursors[asset.asset_key] = cursor + dur + 0.35

        cuts.append(
            VisualCut(
                index=idx,
                asset=asset,
                source_start=round(max(0.0, cursor), 3),
                duration=round(dur, 3),
                motion_mode=idx % 4,
                punch_zoom=bool(front_loaded_hook and idx == 0),
            )
        )
    return cuts


def _render_single_cut(
    cut: VisualCut,
    out_mp4: Path,
    *,
    width: int,
    height: int,
    fps: int = VIDEO_FPS,
) -> Path:
    """Normalize, color-grade, and animate one visual cut with FFmpeg (including punch-zoom when enabled)."""
    assert cut.asset.local_path is not None
    src = cut.asset.local_path
    dur = max(0.5, cut.duration)
    is_target_portrait = height > width

    grade = "eq=contrast=1.08:saturation=1.16:brightness=0.015,setsar=1"
    need_loop = cut.asset.duration < (cut.source_start + dur + 0.25)

    # Determine whether to full-bleed animate or use tasteful blurred ambient fill (for low-res landscape in 9:16)
    use_blurred_frame = (
        is_target_portrait
        and not cut.asset.is_portrait
        and cut.asset.width < 2160
    )

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if need_loop:
        safe_loop_start = min(cut.source_start, max(0.0, cut.asset.duration - 1.0))
        cmd.extend(["-stream_loop", "-1", "-i", str(src)])
        if safe_loop_start > 0.05:
            cmd.extend(["-ss", f"{safe_loop_start:.3f}"])
        cmd.extend(["-t", f"{dur:.3f}"])
    else:
        cmd.extend(["-ss", f"{cut.source_start:.3f}", "-i", str(src), "-t", f"{dur:.3f}"])

    if use_blurred_frame:
        # Tasteful ambient blurred fill + sharp centered foreground lifted above Shorts subtitles
        fg_y = max(120, int((height - int(width * cut.asset.height / max(1, cut.asset.width))) * 0.38))
        filter_complex = (
            f"[0:v]fps={fps},split=2[bg_src][fg_src];"
            f"[bg_src]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=24:4,eq=brightness=-0.10:saturation=1.15[bg];"
            f"[fg_src]scale={width}:-2:flags=fast_bilinear,{grade}[fg];"
            f"[bg][fg]overlay=(W-w)/2:{fg_y}:shortest=1,format=yuv420p[vout]"
        )
        cmd.extend(["-filter_complex", filter_complex, "-map", "[vout]"])
    elif cut.punch_zoom:
        # Real dynamic scroll-stopping punch zoom-in (1.00x -> 1.13x) on hook & climax payoff cuts!
        total_frames = max(1, int(round(dur * fps)))
        vf = (
            f"fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
            f"crop={width}:{height},"
            f"zoompan=z='1+0.13*min(1,on/{total_frames})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps},"
            f"{grade},format=yuv420p"
        )
        cmd.extend(["-vf", vf])
    else:
        # Smooth floating-point crop motion over slightly oversized canvas (fast, silky, zero jitter)
        overscan = 1.10
        ow = int(math.ceil((width * overscan) / 2.0) * 2)
        oh = int(math.ceil((height * overscan) / 2.0) * 2)
        prog = f"min(1\\,max(0\\,t/{dur:.3f}))"
        if cut.motion_mode == 0:
            x_expr = f"(iw-{width})*{prog}"
            y_expr = f"(ih-{height})/2"
        elif cut.motion_mode == 1:
            x_expr = f"(iw-{width})*(1-{prog})"
            y_expr = f"(ih-{height})/2"
        elif cut.motion_mode == 2:
            x_expr = f"(iw-{width})/2"
            y_expr = f"(ih-{height})*{prog}"
        else:
            x_expr = f"(iw-{width})/2"
            y_expr = f"(ih-{height})*(1-{prog})"

        vf = (
            f"fps={fps},"
            f"scale={ow}:{oh}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
            f"crop={width}:{height}:x='{x_expr}':y='{y_expr}',"
            f"{grade},format=yuv420p"
        )
        cmd.extend(["-vf", vf])

    cmd.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-r",
            str(fps),
            str(out_mp4),
        ]
    )
    subprocess.run(cmd, check=True, timeout=180)
    return out_mp4


def _concat_segments(segments: list[Path], out_mp4: Path, work_dir: Path) -> Path:
    concat_file = work_dir / f"{out_mp4.stem}_concat.txt"
    concat_file.write_text("\n".join(f"file '{p.resolve()}'" for p in segments) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(out_mp4),
        ],
        check=True,
        timeout=180,
    )
    return out_mp4


def _finalize_master(
    stitched_video: Path,
    mixed_audio_m4a: Path,
    scrim_png: Path,
    ass_path: Path | None,
    out_mp4: Path,
    *,
    total_duration: float,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Apply smooth readability scrim, burn ASS captions/titles, and mux final AAC audio."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if ass_path and ass_path.is_file():
        escaped = _escape_ffmpeg_filter_path(ass_path)
        filter_complex = f"[0:v][1:v]overlay=0:0:shortest=1,ass='{escaped}',format=yuv420p[vout]"
    else:
        filter_complex = "[0:v][1:v]overlay=0:0:shortest=1,format=yuv420p[vout]"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(stitched_video),
            "-loop",
            "1",
            "-i",
            str(scrim_png),
            "-i",
            str(mixed_audio_m4a),
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "2:a:0",
            "-t",
            f"{total_duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-r",
            str(VIDEO_FPS),
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ],
        check=True,
        timeout=1800,
    )
    return out_mp4


def render_short(
    clips: list[MediaAsset],
    mixed_audio_m4a: Path,
    ass_path: Path,
    out_mp4: Path,
    work_dir: Path,
    *,
    total_duration: float,
    width: int = SHORTS_WIDTH,
    height: int = SHORTS_HEIGHT,
    beat_clips: list[list[MediaAsset]] | None = None,
    beat_specs: list[tuple[float, float, bool]] | None = None,
) -> Path:
    """Render a crisp, fast-paced 9:16 viral Short with beat-synced visuals, punch-zoom, and karaoke ASS."""
    seg_dir = work_dir / "segments_short"
    seg_dir.mkdir(parents=True, exist_ok=True)

    if beat_clips and beat_specs:
        cuts = plan_beat_synced_cuts(
            beat_clips,
            beat_specs,
            total_duration + 0.12,
            work_dir=work_dir,
            max_shot_seconds=2.15,
        )
    else:
        ranked_clips = rank_clips_for_hook(clips, work_dir)
        cuts = plan_visual_cuts(
            ranked_clips,
            total_duration + 0.12,
            target_cut_seconds=2.25,
            min_cut_seconds=1.65,
            front_loaded_hook=True,
        )
    segments: list[Path] = []
    for cut in cuts:
        seg_path = seg_dir / f"seg_{cut.index:03d}.mp4"
        _render_single_cut(cut, seg_path, width=width, height=height)
        segments.append(seg_path)

    stitched = work_dir / "short_stitched.mp4"
    _concat_segments(segments, stitched, work_dir)

    scrim = create_gradient_scrim(
        work_dir / "short_scrim.png",
        width=width,
        height=height,
        top_ratio=0.14,
        top_alpha=105,
        bottom_ratio=0.38,
        bottom_alpha=165,
    )
    return _finalize_master(
        stitched,
        mixed_audio_m4a,
        scrim,
        ass_path,
        out_mp4,
        total_duration=total_duration,
        crf=20,
        preset="veryfast",
    )


def render_documentary(
    clips: list[MediaAsset],
    mixed_audio_m4a: Path,
    chapters_ass: Path,
    out_mp4: Path,
    work_dir: Path,
    *,
    total_duration: float,
    width: int = DOC_WIDTH,
    height: int = DOC_HEIGHT,
) -> Path:
    """Render a cinematic 16:9 1080p Documentary (7.5-10s shots + lower-third chapter titles)."""
    seg_dir = work_dir / "segments_doc"
    seg_dir.mkdir(parents=True, exist_ok=True)

    ranked_clips = rank_clips_for_hook(clips, work_dir, prefer_portrait=False)
    cuts = plan_visual_cuts(ranked_clips, total_duration + 0.25, target_cut_seconds=8.5, min_cut_seconds=5.5)
    segments: list[Path] = []
    for cut in cuts:
        seg_path = seg_dir / f"doc_seg_{cut.index:03d}.mp4"
        _render_single_cut(cut, seg_path, width=width, height=height)
        segments.append(seg_path)

    stitched = work_dir / "doc_stitched.mp4"
    _concat_segments(segments, stitched, work_dir)

    scrim = create_gradient_scrim(
        work_dir / "doc_scrim.png",
        width=width,
        height=height,
        top_ratio=0.08,
        top_alpha=55,
        bottom_ratio=0.24,
        bottom_alpha=125,
    )
    return _finalize_master(
        stitched,
        mixed_audio_m4a,
        scrim,
        chapters_ass,
        out_mp4,
        total_duration=total_duration,
        crf=21,
        preset="superfast",
    )


def verify_rendered_video(
    mp4_path: Path,
    *,
    expected_width: int,
    expected_height: int,
    min_duration: float,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Run automated quality control on a rendered MP4 (streams, dimensions, visuals, loudness)."""
    if not mp4_path.is_file() or mp4_path.stat().st_size < 100_000:
        raise RuntimeError(f"Rendered video missing or suspiciously small: {mp4_path}")

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height:format=duration,size",
            "-of",
            "json",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    info = json.loads(probe.stdout)
    streams = info.get("streams") or []
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v_stream or not a_stream:
        raise RuntimeError(f"Missing video or audio stream in {mp4_path}")

    width = int(v_stream.get("width") or 0)
    height = int(v_stream.get("height") or 0)
    duration = float(info.get("format", {}).get("duration") or 0)
    if width != expected_width or height != expected_height:
        raise RuntimeError(f"Unexpected dimensions {width}x{height} (expected {expected_width}x{expected_height})")
    if duration < min_duration:
        raise RuntimeError(f"Video too short: {duration:.2f}s < {min_duration:.2f}s")

    # Visual non-black / non-solid frame check across 5 checkpoints.
    # Retries ±0.75s around a dark frame so split-second transitions or a single dark night/campfire
    # shot never false-positive, while genuinely blank/black/corrupted renders still fail reliably.
    qc_dir = (work_dir or mp4_path.parent) / "qc_frames"
    qc_dir.mkdir(parents=True, exist_ok=True)
    frame_stats: list[dict[str, float]] = []
    dark_pcts: list[int] = []
    for pct in (0.15, 0.35, 0.50, 0.65, 0.85):
        base_ts = max(0.2, min(duration - 0.2, duration * pct))
        pct_label = int(pct * 100)
        metrics = _probe_frame_metrics(mp4_path, base_ts, qc_dir / f"qc_{pct_label}.png")
        if metrics is None:
            raise RuntimeError(f"Could not decode QC frame at {pct_label}%")
        mean_lum, std_lum, _ = metrics

        if mean_lum < 3.5 or std_lum < 3.5:
            best_mean, best_std = mean_lum, std_lum
            for offset_idx, delta in enumerate((-0.75, 0.75)):
                alt_ts = max(0.2, min(duration - 0.2, base_ts + delta))
                if abs(alt_ts - base_ts) < 0.1:
                    continue
                alt = _probe_frame_metrics(mp4_path, alt_ts, qc_dir / f"qc_{pct_label}_alt{offset_idx}.png")
                if alt is None:
                    continue
                if (alt[1] >= 3.5 and alt[0] > best_mean) or (best_std < 3.5 and alt[1] > best_std):
                    best_mean, best_std = alt[0], alt[1]
            mean_lum, std_lum = best_mean, best_std

        frame_stats.append({"pct": pct, "mean": round(mean_lum, 1), "stddev": round(std_lum, 1)})
        if mean_lum < 3.5 or std_lum < 3.5:
            dark_pcts.append(pct_label)

    avg_mean = sum(f["mean"] for f in frame_stats) / len(frame_stats)
    max_std = max(f["stddev"] for f in frame_stats)
    if max_std < 4.0 or avg_mean < 8.0 or len(dark_pcts) >= 2:
        raise RuntimeError(
            f"Video failed visual QC (dark_pcts={dark_pcts}, avg_mean={avg_mean:.1f}, "
            f"max_std={max_std:.1f}, stats={frame_stats})"
        )
    if dark_pcts:
        print(
            f"[QC] Warning: isolated dark frame at {dark_pcts[0]}% "
            f"(other {len(frame_stats) - len(dark_pcts)}/{len(frame_stats)} frames healthy, avg_mean={avg_mean:.1f}); allowing night scene",
            flush=True,
        )

    # Audio loudness verification via volumedetect
    vol_proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(mp4_path),
            "-af",
            "volumedetect",
            "-vn",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    match = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", vol_proc.stderr)
    mean_db = float(match.group(1)) if match else -99.0
    if mean_db < -38.0:
        raise RuntimeError(f"Rendered audio is nearly silent (mean_volume={mean_db:.1f} dB)")

    return {
        "width": width,
        "height": height,
        "duration": round(duration, 2),
        "video_codec": v_stream.get("codec_name"),
        "audio_codec": a_stream.get("codec_name"),
        "mean_volume_db": mean_db,
        "size_mb": round(mp4_path.stat().st_size / (1024 * 1024), 2),
        "frame_stats": frame_stats,
    }
