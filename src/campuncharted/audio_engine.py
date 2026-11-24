"""Edge-TTS WordBoundary synthesizer, tight silence trimmer, and procedural stereo nature ambience."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import edge_tts
import numpy as np

from .config import (
    DOC_CHAPTER_GAP_SECONDS,
    DOC_TTS_PITCH,
    DOC_TTS_RATE,
    SHORTS_BEAT_GAP_SECONDS,
    SHORTS_TTS_PITCH,
    SHORTS_TTS_RATE,
)


@dataclass
class WordTiming:
    word: str
    start: float  # seconds on global video timeline
    end: float    # seconds on global video timeline


@dataclass
class BeatAudio:
    index: int
    text: str
    audio_path: Path
    start: float
    duration: float
    words: list[WordTiming]


def probe_audio_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return max(0.05, float(proc.stdout.strip()))


async def _synthesize_one(
    text: str,
    out_mp3: Path,
    voices: tuple[str, ...],
    *,
    rate: str,
    pitch: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Synthesize text via Edge-TTS using WordBoundary events, falling back smoothly across voices."""
    last_error: Exception | None = None
    for voice in voices:
        for attempt in range(2):
            try:
                if out_mp3.exists():
                    out_mp3.unlink()
                comm = edge_tts.Communicate(
                    text,
                    voice,
                    rate=rate,
                    pitch=pitch,
                    boundary="WordBoundary",
                )
                boundaries: list[dict[str, Any]] = []
                audio_bytes = bytearray()
                async for chunk in comm.stream():
                    ctype = chunk.get("type")
                    if ctype == "audio":
                        audio_bytes.extend(chunk["data"])
                    elif ctype in ("WordBoundary", "SentenceBoundary"):
                        boundaries.append(
                            {
                                "type": ctype,
                                "offset_sec": float(chunk.get("offset", 0)) / 10_000_000.0,
                                "duration_sec": float(chunk.get("duration", 0)) / 10_000_000.0,
                                "text": str(chunk.get("text", "")),
                            }
                        )
                if len(audio_bytes) < 512:
                    raise RuntimeError("Empty TTS output")
                out_mp3.write_bytes(bytes(audio_bytes))
                return voice, boundaries
            except Exception as err:
                last_error = err
                await asyncio.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Edge-TTS failed across all voices: {last_error}")


def _trim_silence_and_convert(src_mp3: Path, dst_wav: Path) -> tuple[float, float]:
    """Trim leading/trailing dead silence tightly so Shorts have zero awkward pauses.

    Returns (new_duration_sec, estimated_leading_trim_sec).
    """
    raw_dur = probe_audio_duration(src_mp3)
    # Trim leading silence gently and trailing silence tightly while converting to 44.1kHz stereo WAV
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_mp3),
            "-af",
            "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-42dB,"
            "areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-42dB,areverse,"
            "highpass=f=75,equalizer=f=2800:width_type=o:width=1.5:g=1.8",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(dst_wav),
        ],
        check=True,
        timeout=45,
    )
    new_dur = probe_audio_duration(dst_wav)
    lead_shift = max(0.0, min(0.12, (raw_dur - new_dur) * 0.35))
    return new_dur, lead_shift


def build_word_timings(
    text: str,
    boundaries: list[dict[str, Any]],
    *,
    timeline_offset: float,
    clip_duration: float,
    lead_shift: float = 0.0,
) -> list[WordTiming]:
    """Map Edge-TTS WordBoundary events onto the trimmed beat timeline (with proportional fallback)."""
    word_events = [b for b in boundaries if b.get("type") == "WordBoundary" and b.get("text", "").strip()]
    timings: list[WordTiming] = []
    if word_events:
        max_start = max(0.0, clip_duration - 0.08)
        for ev in word_events:
            s = min(max_start, max(0.0, ev["offset_sec"] - lead_shift))
            d = max(0.08, ev["duration_sec"])
            e = min(clip_duration, s + d)
            if e <= s:
                e = min(clip_duration, s + 0.08)
            timings.append(WordTiming(word=ev["text"].strip(), start=timeline_offset + s, end=timeline_offset + e))
        # Ensure strictly monotonic non-overlapping times
        for i in range(len(timings) - 1):
            if timings[i].end > timings[i + 1].start:
                timings[i].end = max(timings[i].start + 0.04, timings[i + 1].start)
        return timings

    # Clean proportional fallback if voice emitted only SentenceBoundary or synthetic audio was used
    words = [w for w in str(text).split() if w.strip()]
    if not words:
        return []
    weights = [max(2, len(w)) for w in words]
    total_w = float(sum(weights))
    cursor = timeline_offset
    for word, weight in zip(words, weights):
        dur = clip_duration * (weight / total_w)
        timings.append(WordTiming(word=word, start=cursor, end=cursor + dur))
        cursor += dur
    return timings


def synthesize_narration_beats(
    texts: list[str],
    work_dir: Path,
    *,
    voices: tuple[str, ...],
    rate: str = SHORTS_TTS_RATE,
    pitch: str = SHORTS_TTS_PITCH,
    gap_seconds: float = SHORTS_BEAT_GAP_SECONDS,
) -> tuple[Path, list[BeatAudio], float]:
    """Synthesize all narration beats with consistent voice, trim silence, and concat into narration.wav."""
    audio_dir = work_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    async def _run_all() -> list[tuple[Path, list[dict[str, Any]]]]:
        results: list[tuple[Path, list[dict[str, Any]]]] = []
        locked_voice: tuple[str, ...] = voices
        for idx, text in enumerate(texts):
            mp3 = audio_dir / f"beat_{idx:02d}.mp3"
            used_voice, bounds = await _synthesize_one(text, mp3, locked_voice, rate=rate, pitch=pitch)
            locked_voice = (used_voice,) + tuple(v for v in voices if v != used_voice)
            results.append((mp3, bounds))
        return results

    raw_results = asyncio.run(_run_all())

    beats: list[BeatAudio] = []
    concat_parts: list[Path] = []
    gap_wav = audio_dir / "gap.wav"
    _write_silence_wav(gap_wav, gap_seconds)

    timeline = 0.0
    for idx, (text, (mp3, bounds)) in enumerate(zip(texts, raw_results)):
        wav = audio_dir / f"beat_{idx:02d}.wav"
        dur, lead_shift = _trim_silence_and_convert(mp3, wav)
        words = build_word_timings(
            text,
            bounds,
            timeline_offset=timeline,
            clip_duration=dur,
            lead_shift=lead_shift,
        )
        beats.append(BeatAudio(index=idx, text=text, audio_path=wav, start=timeline, duration=dur, words=words))
        concat_parts.append(wav)
        timeline += dur
        if idx < len(texts) - 1 and gap_seconds > 0.01:
            concat_parts.append(gap_wav)
            timeline += gap_seconds

    concat_list = audio_dir / "narration_concat.txt"
    concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in concat_parts) + "\n", encoding="utf-8")
    narration_wav = audio_dir / "narration_full.wav"
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
            str(concat_list),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(narration_wav),
        ],
        check=True,
        timeout=120,
    )
    actual_dur = probe_audio_duration(narration_wav)
    return narration_wav, beats, actual_dur


def _write_silence_wav(path: Path, seconds: float, sample_rate: int = 44100) -> None:
    n = max(1, int(sample_rate * max(0.01, seconds)))
    data = np.zeros((n, 2), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def generate_nature_soundscape(
    out_wav: Path,
    duration_seconds: float,
    *,
    mood: str = "ALPINE_WIND",
    include_hook_swell: bool = True,
    loop_snappy: bool = False,
    beat_timestamps: list[float] | None = None,
    sample_rate: int = 44100,
) -> Path:
    """Generate a clean, pleasant, zero-dependency stereo nature soundscape bed with optional beat-synced SFX.

    Moods:
    - ALPINE_WIND: soft stereo mountain breeze + warm open fifth ambient pad
    - FOREST_RAIN: gentle stereo rain patter + warm woody low-end
    - CAMPFIRE_NIGHT: cozy soft night breeze + subtle organic wood crackles + warm drone
    - CANYON_BREEZE: airy desert wind + warm resonant pad
    - EPIC_DISCOVERY: lush cinematic warm harmonic bed + gentle nature breeze
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    n = max(sample_rate, int(sample_rate * (duration_seconds + 0.5)))
    seed_hex = hashlib.sha256(f"{mood}:{int(duration_seconds)}".encode("utf-8")).hexdigest()[:8]
    rng = np.random.default_rng(int(seed_hex, 16))
    t = np.linspace(0.0, n / sample_rate, n, endpoint=False, dtype=np.float32)

    # Smooth pink/brown noise foundation via FFT spectral shaping (fast & clean in numpy)
    def colored_noise(beta: float) -> np.ndarray:
        white = rng.standard_normal(n).astype(np.float32)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        freqs[0] = 1.0
        scale = 1.0 / (freqs ** (beta / 2.0))
        # Band-limit cleanly between 60 Hz and 6500 Hz
        scale *= np.where(freqs < 60.0, freqs / 60.0, 1.0)
        scale *= np.exp(-((freqs / 4800.0) ** 2))
        shaped = np.fft.irfft(spec * scale, n=n).astype(np.float32)
        peak = float(np.max(np.abs(shaped))) or 1.0
        return shaped / peak

    left_noise = colored_noise(1.55 if mood != "FOREST_RAIN" else 1.05)
    right_noise = colored_noise(1.55 if mood != "FOREST_RAIN" else 1.05)

    # Slow organic stereo wind/rain breathing envelope
    mod_l = 0.65 + 0.35 * np.sin(2.0 * math.pi * 0.11 * t + 0.4)
    mod_r = 0.65 + 0.35 * np.sin(2.0 * math.pi * 0.09 * t + 2.1)
    left = left_noise * mod_l * (0.28 if mood == "FOREST_RAIN" else 0.18)
    right = right_noise * mod_r * (0.28 if mood == "FOREST_RAIN" else 0.18)

    # Warm, restful harmonic ambient pad (D major / A major cinema chord — NEVER horror!)
    chord_hz = {
        "ALPINE_WIND": (146.83, 220.00, 293.66, 369.99),     # D3, A3, D4, F#4
        "FOREST_RAIN": (130.81, 196.00, 261.63, 329.63),     # C3, G3, C4, E4
        "CAMPFIRE_NIGHT": (110.00, 164.81, 220.00, 277.18),  # A2, E3, A3, C#4
        "CANYON_BREEZE": (146.83, 220.00, 329.63, 440.00),   # D3, A3, E4, A4 (open sus2)
        "EPIC_DISCOVERY": (130.81, 196.00, 293.66, 392.00),  # C3, G3, D4, G4 (cinematic open)
    }.get(mood, (146.83, 220.00, 293.66, 369.99))

    pad_l = np.zeros(n, dtype=np.float32)
    pad_r = np.zeros(n, dtype=np.float32)
    for i, hz in enumerate(chord_hz):
        swell = 0.55 + 0.45 * np.sin(2.0 * math.pi * (0.045 + 0.015 * i) * t + i * 1.1)
        amp = 0.085 / (1.0 + 0.35 * i)
        pad_l += (np.sin(2.0 * math.pi * hz * t) * swell * amp).astype(np.float32)
        pad_r += (np.sin(2.0 * math.pi * (hz * 1.002) * t + 0.5) * swell * amp).astype(np.float32)

    left += pad_l
    right += pad_r

    # Subtle cozy campfire crackles for CAMPFIRE_NIGHT
    if mood == "CAMPFIRE_NIGHT":
        crackle_count = max(8, int(duration_seconds * 4.5))
        positions = rng.integers(0, max(1, n - 400), size=crackle_count)
        for pos in positions:
            length = int(rng.integers(45, 160))
            decay = np.exp(-np.linspace(0.0, 5.5, length, dtype=np.float32))
            burst = rng.standard_normal(length).astype(np.float32) * decay * float(rng.uniform(0.08, 0.22))
            pan = float(rng.uniform(0.25, 0.75))
            left[pos : pos + length] += burst * pan
            right[pos : pos + length] += burst * (1.0 - pan)

    # Subtle cinematic momentum pulse (~108 BPM muted low pulse) exclusively for viral Shorts
    if loop_snappy and duration_seconds >= 5.0:
        step_sec = 60.0 / 108.0
        pulse_t = step_sec
        pulse_idx = 0
        while pulse_t < duration_seconds - 0.25:
            pos = int(pulse_t * sample_rate)
            p_len = min(n - pos, int(sample_rate * 0.14))
            if p_len > 100:
                pt = np.linspace(0.0, p_len / sample_rate, p_len, dtype=np.float32)
                base_hz = 74.0 if (pulse_idx % 2 == 0) else 62.0
                thump = np.sin(2.0 * math.pi * (base_hz - 24.0 * pt) * pt) * np.exp(-pt * 24.0) * 0.13
                left[pos : pos + p_len] += thump
                right[pos : pos + p_len] += thump
            pulse_t += step_sec
            pulse_idx += 1

    # Crisp sub-impact + airy whoosh at t=0.0s, beat-synced transition whooshes, and climax riser
    if include_hook_swell:
        hit_len = min(n, int(sample_rate * 0.48))
        ht = np.linspace(0.0, hit_len / sample_rate, hit_len, dtype=np.float32)
        sub = np.sin(2.0 * math.pi * (96.0 - 46.0 * ht) * ht) * np.exp(-ht * 7.2) * 0.36
        whoosh = left_noise[:hit_len] * np.sin(math.pi * ht / max(0.01, ht[-1])) * 0.24
        left[:hit_len] += sub + whoosh
        right[:hit_len] += sub + whoosh

        # Add tactile airy transition whooshes synced to spoken beat cuts (keeps retention sharp every ~2-3s)
        valid_beats = sorted(
            ts for ts in (beat_timestamps or []) if 0.9 <= float(ts) <= max(0.9, duration_seconds - 0.6)
        )
        climax_ts = valid_beats[-2] if len(valid_beats) >= 3 else (duration_seconds * 0.62 if duration_seconds >= 12.0 else None)

        for idx_b, ts in enumerate(valid_beats):
            if climax_ts is not None and abs(ts - climax_ts) < 0.2:
                continue
            w_len = int(sample_rate * 0.16)
            start_i = max(0, int((ts - 0.06) * sample_rate))
            end_i = min(n, start_i + w_len)
            actual = end_i - start_i
            if actual > 120:
                wt = np.linspace(0.0, actual / sample_rate, actual, dtype=np.float32)
                env = np.maximum(0.0, np.sin(math.pi * wt / max(0.01, float(wt[-1])))) ** 1.4
                pan_l = 0.65 if (idx_b % 2 == 0) else 0.35
                sweep = left_noise[start_i:end_i] * env * 0.16
                snap = np.sin(2.0 * math.pi * (140.0 - 65.0 * wt) * wt) * np.exp(-wt * 22.0) * 0.10
                left[start_i:end_i] += sweep * pan_l + snap
                right[start_i:end_i] += sweep * (1.0 - pan_l) + snap

        # Climax/payoff tension riser into crisp hit right before the big reveal
        if climax_ts is not None and climax_ts > 1.2:
            riser_sec = 0.42
            r_end = min(n, int(climax_ts * sample_rate))
            r_start = max(0, r_end - int(riser_sec * sample_rate))
            r_len = r_end - r_start
            if r_len > 200:
                rt = np.linspace(0.0, r_len / sample_rate, r_len, dtype=np.float32)
                r_env = (rt / max(0.01, float(rt[-1]))) ** 2.0
                riser_tone = np.sin(2.0 * math.pi * (180.0 + 360.0 * (rt / max(0.01, float(rt[-1])))) * rt) * r_env * 0.11
                riser_air = left_noise[r_start:r_end] * r_env * 0.20
                left[r_start:r_end] += riser_tone + riser_air
                right[r_start:r_end] += riser_tone + riser_air

            tw_pos = r_end
            tw_len = min(n - tw_pos, int(sample_rate * 0.36))
            if tw_len > 200:
                wt = np.linspace(0.0, tw_len / sample_rate, tw_len, dtype=np.float32)
                accent = np.sin(2.0 * math.pi * (86.0 - 36.0 * wt) * wt) * np.exp(-wt * 7.8) * 0.28
                left[tw_pos : tw_pos + tw_len] += accent
                right[tw_pos : tw_pos + tw_len] += accent

    # Seamless loop-friendly micro-fade for Shorts vs long cinematic fade for Documentaries
    fade_in = min(n, int(sample_rate * (0.04 if loop_snappy else 0.30)))
    fade_out = min(n, int(sample_rate * (0.03 if loop_snappy else 1.20)))
    left[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    right[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    left[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    right[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    stereo = np.nan_to_num(np.stack([left, right], axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(stereo))) or 1.0
    stereo = (stereo / peak) * 0.45
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out_wav


def mix_narration_and_ambience(
    narration_wav: Path,
    ambience_wav: Path,
    out_m4a: Path,
    *,
    duration_seconds: float,
    ambience_volume: float = 0.24,
) -> Path:
    """Broadcast-grade sidechain ducking + EBU R128 loudness normalization (-14 LUFS)."""
    out_m4a.parent.mkdir(parents=True, exist_ok=True)
    # Split narration into main voice + sidechain trigger so ambience ducks cleanly during speech
    filter_complex = (
        f"[0:a]asplit=2[voice_main][voice_sc];"
        f"[1:a]atrim=0:{duration_seconds:.3f},volume={ambience_volume:.3f}[amb];"
        f"[amb][voice_sc]sidechaincompress=threshold=0.018:ratio=5:attack=25:release=320:makeup=1[ducked_amb];"
        f"[voice_main]acompressor=threshold=-18dB:ratio=2.8:attack=15:release=220:makeup=2.5[comp_voice];"
        f"[comp_voice][ducked_amb]amix=inputs=2:duration=first:dropout_transition=0.4:normalize=0,"
        f"loudnorm=I=-14:TP=-1.0:LRA=9[aout]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(narration_wav),
            "-i",
            str(ambience_wav),
            "-filter_complex",
            filter_complex,
            "-map",
            "[aout]",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            str(out_m4a),
        ],
        check=True,
        timeout=180,
    )
    return out_m4a
