"""Offline end-to-end FFmpeg + Pillow + Numpy rendering & QC integration tests."""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np

from campuncharted.audio_engine import (
    BeatAudio,
    build_word_timings,
    generate_nature_soundscape,
    mix_narration_and_ambience,
)
from campuncharted.media_engine import MediaAsset, probe_video
from campuncharted.subtitle_engine import (
    generate_documentary_chapters_ass,
    generate_shorts_karaoke_ass,
)
from campuncharted.thumbnail_engine import extract_best_frame, generate_thumbnail
from campuncharted.video_engine import (
    render_documentary,
    render_short,
    verify_rendered_video,
)


def _make_synthetic_clip(out_mp4: Path, *, width: int, height: int, duration: float, filter_src: str) -> MediaAsset:
    """Create a rich, textured synthetic video clip via FFmpeg (no external downloads needed)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"{filter_src}=size={width}x{height}:rate=30:duration={duration:.2f}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ],
        check=True,
        timeout=60,
    )
    probed = probe_video(out_mp4)
    assert probed is not None
    return MediaAsset(
        asset_key=f"test_{out_mp4.stem}",
        provider="Synthetic",
        asset_id=out_mp4.stem,
        download_url="",
        page_url="",
        title=out_mp4.stem,
        author="Test",
        license_name="CC0",
        width=probed["width"],
        height=probed["height"],
        duration=probed["duration"],
        local_path=out_mp4,
    )


def _make_synthetic_narration(out_wav: Path, duration: float, sample_rate: int = 44100) -> Path:
    """Create speech-band modulated audio so sidechain ducking and volumedetect QC exercise real signal."""
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    # Syllabic amplitude envelope (~4 Hz speech cadence) with warm voice harmonics
    envelope = (0.5 + 0.5 * np.sin(2.0 * math.pi * 3.8 * t)) * 0.45
    voice = (
        np.sin(2.0 * math.pi * 165.0 * t) * 0.50
        + np.sin(2.0 * math.pi * 330.0 * t) * 0.30
        + np.sin(2.0 * math.pi * 660.0 * t) * 0.15
    ) * envelope
    stereo = np.stack([voice, voice], axis=1)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out_wav


def test_e2e_short_render_karaoke_and_thumbnail(tmp_path: Path) -> None:
    # Mix a native portrait clip AND a 1080p landscape fallback clip to exercise both render paths!
    portrait_clip = _make_synthetic_clip(
        tmp_path / "clip_portrait.mp4",
        width=1080,
        height=1920,
        duration=5.0,
        filter_src="testsrc2",
    )
    landscape_clip = _make_synthetic_clip(
        tmp_path / "clip_landscape.mp4",
        width=1920,
        height=1080,
        duration=5.0,
        filter_src="smptehdbars",
    )

    total_dur = 7.2
    narration_wav = _make_synthetic_narration(tmp_path / "narr.wav", total_dur)
    ambience_wav = generate_nature_soundscape(
        tmp_path / "ambience.wav",
        total_dur + 0.5,
        mood="CAMPFIRE_NIGHT",
        include_hook_swell=True,
    )
    mixed_m4a = mix_narration_and_ambience(
        narration_wav,
        ambience_wav,
        tmp_path / "mixed.m4a",
        duration_seconds=total_dur,
    )

    lines = [
        "Never pitch your tent inside a dry desert canyon riverbed.",
        "Storms twenty miles upstream send silent walls of water rushing down.",
    ]
    beats: list[BeatAudio] = []
    cursor = 0.0
    for idx, line in enumerate(lines):
        dur = total_dur / len(lines)
        words = build_word_timings(line, [], timeline_offset=cursor, clip_duration=dur - 0.1)
        beats.append(BeatAudio(index=idx, text=line, audio_path=narration_wav, start=cursor, duration=dur, words=words))
        cursor += dur

    ass_path = generate_shorts_karaoke_ass(beats, tmp_path / "captions.ass", pill_label="CANYON SAFETY")
    out_mp4 = tmp_path / "rendered_short.mp4"
    render_short(
        [portrait_clip, landscape_clip],
        mixed_m4a,
        ass_path,
        out_mp4,
        tmp_path / "work",
        total_duration=total_dur,
    )

    qc = verify_rendered_video(
        out_mp4,
        expected_width=1080,
        expected_height=1920,
        min_duration=6.5,
        work_dir=tmp_path / "work",
    )
    assert qc["video_codec"] == "h264"
    assert qc["audio_codec"] == "aac"
    assert qc["mean_volume_db"] > -30.0

    best_frame = extract_best_frame(out_mp4, tmp_path / "work", sample_count=4)
    thumb = generate_thumbnail(
        best_frame,
        tmp_path / "short_thumb.jpg",
        hook_text="NEVER CAMP HERE",
        sub_label="CANYON SAFETY",
        format_type="short",
    )
    assert thumb.is_file()
    assert 20_000 < thumb.stat().st_size < 1_900_000


def test_e2e_documentary_render_and_thumbnail(tmp_path: Path) -> None:
    clip1 = _make_synthetic_clip(
        tmp_path / "doc_clip1.mp4",
        width=1280,
        height=720,
        duration=6.0,
        filter_src="testsrc2",
    )
    total_dur = 5.5
    narration_wav = _make_synthetic_narration(tmp_path / "doc_narr.wav", total_dur)
    ambience_wav = generate_nature_soundscape(
        tmp_path / "doc_amb.wav",
        total_dur + 0.5,
        mood="ALPINE_WIND",
        include_hook_swell=False,
    )
    mixed_m4a = mix_narration_and_ambience(
        narration_wav,
        ambience_wav,
        tmp_path / "doc_mixed.m4a",
        duration_seconds=total_dur,
    )
    chapters_ass = generate_documentary_chapters_ass([(0.0, total_dur, 1, "Into The Glaciers")], tmp_path / "doc.ass")
    out_mp4 = tmp_path / "rendered_doc.mp4"
    render_documentary(
        [clip1],
        mixed_m4a,
        chapters_ass,
        out_mp4,
        tmp_path / "doc_work",
        total_duration=total_dur,
    )
    qc = verify_rendered_video(
        out_mp4,
        expected_width=1920,
        expected_height=1080,
        min_duration=5.0,
        work_dir=tmp_path / "doc_work",
    )
    assert qc["width"] == 1920 and qc["height"] == 1080

    best_frame = extract_best_frame(out_mp4, tmp_path / "doc_work", sample_count=3)
    thumb = generate_thumbnail(
        best_frame,
        tmp_path / "doc_thumb.jpg",
        hook_text="NEON GLACIER LAKES",
        sub_label="BANFF NATIONAL PARK",
        format_type="documentary",
    )
    assert thumb.is_file()
    assert 20_000 < thumb.stat().st_size < 1_900_000
