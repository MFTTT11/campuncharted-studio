"""Central configuration for CampUncharted Studio v2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_CHANNEL_ID = "UCwzbmfB6P-s7RrZqHA1JCqw"
CHANNEL_HANDLE = "@CampUncharted"
CHANNEL_NAME = "CampUncharted"

# Shorts & Documentary production targets
SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
DOC_WIDTH = 1920
DOC_HEIGHT = 1080
VIDEO_FPS = 30

SHORTS_DAILY_LIMIT = 3
DOC_WEEKLY_LIMIT = 2

# Peak global UTC publishing slots (US morning/lunch/evening + EU afternoon/evening)
SHORTS_UTC_SLOTS = ((13, 15), (18, 15), (23, 15))
DOC_UTC_SLOTS = ((16, 0),)

# Multi-language subtitles & localized titles (priority order)
SUBTITLE_LANGUAGES = {
    "en": "English",
    "es": "Español",
    "tr": "Türkçe",
    "de": "Deutsch",
    "fr": "Français",
}

# Daily quota cap for caption/localization operations (out of 10,000 YouTube units/day)
# Keeps 8,200+ units always reserved for 3 Shorts (4,800) + 1 Documentary (1,600) + thumbnails.
DAILY_LANGUAGE_QUOTA_BUDGET = 1800

# High-quality Gemini model preference (active 3.x models first; 2.5 is retired for new keys!)
GEMINI_PREFERRED_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.0-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
)

# High-retention natural Edge-TTS narration voices
SHORTS_VOICES_EN = (
    "en-US-AndrewMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-US-ChristopherNeural",
    "en-GB-RyanNeural",
)
DOCUMENTARY_VOICES_EN = (
    "en-US-BrianMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-GuyNeural",
)

# Shorts narration pacing: punchy, confident, zero dead air for >100% loop retention
SHORTS_TTS_RATE = "+11%"
SHORTS_TTS_PITCH = "+0Hz"
SHORTS_BEAT_GAP_SECONDS = 0.06

# Documentary narration pacing: warm, cinematic, steady
DOC_TTS_RATE = "+0%"
DOC_TTS_PITCH = "-1Hz"
DOC_CHAPTER_GAP_SECONDS = 0.65

# Initial series rotation weights (dynamically updated by analytics_engine)
DEFAULT_SERIES_WEIGHTS = {
    "CAMP_MISTAKES": 0.30,
    "WILD_STAYS": 0.25,
    "GEAR_AND_HACKS": 0.20,
    "EXTREME_WEATHER_NATURE": 0.15,
    "TURKEY_AND_EURASIA_HIDDEN": 0.10,
}

# YouTube category IDs
CATEGORY_TRAVEL_EVENTS = "19"
CATEGORY_EDUCATION = "27"

USER_AGENT = "CampUnchartedStudio/2.0 (https://youtube.com/@CampUncharted)"


@dataclass(frozen=True)
class StudioPaths:
    repo_root: Path
    state_dir: Path
    work_dir: Path
    output_dir: Path
    cache_dir: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "StudioPaths":
        base = Path(root or os.environ.get("CAMP_REPO_ROOT") or Path.cwd()).resolve()
        return cls(
            repo_root=base,
            state_dir=base / "state",
            work_dir=base / "work_tmp",
            output_dir=base / "build_output",
            cache_dir=base / "cache",
        )

    def ensure_dirs(self) -> None:
        for directory in (self.state_dir, self.work_dir, self.output_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ApiCredentials:
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", "").strip())
    pexels_api_key: str = field(default_factory=lambda: os.environ.get("PEXELS_API_KEY", "").strip())
    pixabay_api_key: str = field(default_factory=lambda: os.environ.get("PIXABAY_API_KEY", "").strip())
    coverr_api_key: str = field(default_factory=lambda: os.environ.get("COVERR_API_KEY", "").strip())
    youtube_token_json: str = field(default_factory=lambda: os.environ.get("YOUTUBE_TOKEN_JSON", "").strip())
    youtube_client_secret_json: str = field(
        default_factory=lambda: os.environ.get("YOUTUBE_CLIENT_SECRET_JSON", "").strip()
    )
    expected_channel_id: str = field(
        default_factory=lambda: os.environ.get("YOUTUBE_CHANNEL_ID", EXPECTED_CHANNEL_ID).strip()
    )
