"""Unit tests for CampUncharted Studio v2 core logic, guardrails, and safety locks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.oauth2.credentials import Credentials

from campuncharted.analytics_engine import rebalance_series_weights
from campuncharted.audio_engine import BeatAudio, WordTiming, build_word_timings
from campuncharted.config import DEFAULT_SERIES_WEIGHTS, EXPECTED_CHANNEL_ID
from campuncharted.content_engine import (
    DocumentaryChapter,
    DocumentaryScript,
    ShortBeat,
    ShortScript,
    build_documentary_description,
    build_short_description,
    validate_and_polish_short_script,
)
from campuncharted.media_engine import sanitize_query
from campuncharted.state_store import StateStore, content_hash, normalize_key
from campuncharted.subtitle_engine import (
    beats_to_srt_cues,
    detect_bold_font_path,
    format_ass_time,
    format_srt_time,
    generate_documentary_chapters_ass,
    generate_shorts_karaoke_ass,
    group_words_into_bursts,
    write_srt,
)
from campuncharted.topic_catalog import (
    DOCUMENTARY_TOPICS,
    SHORT_TOPICS,
    pick_next_documentary_topic,
    pick_next_short_topic,
)
from campuncharted.youtube_engine import (
    _build_resumable_http,
    record_reserved_slot,
    reserve_publish_slot,
    verify_channel_ownership,
)


def test_normalize_key_handles_turkish_and_diacritics() -> None:
    assert normalize_key("Lençóis Maranhenses, Brazil") == "lencois maranhenses brazil"
    assert normalize_key("Borçka Karagöl, Türkiye") == "borcka karagol turkey"
    assert normalize_key("Sørvágsvatn, Faroe Islands") == "sorvagsvatn faroe islands"


def test_state_store_dedup_and_persistence(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    store.record_uploaded_location("Svalbard, Norway")
    assert store.is_location_uploaded("Svalbard, Norway")
    assert store.is_location_uploaded("svalbard")
    assert not store.is_location_uploaded("Banff National Park")

    store.record_used_media_ids(["pex_123", "pix_456"])
    assert store.get_used_media_ids() == {"pex_123", "pix_456"}

    digest = content_hash(["topic1", "Why Never Camp Here"])
    store.record_content_hash(digest)
    assert digest in store.get_content_hashes()


def test_series_rotation_avoids_back_to_back_repeats(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    first = pick_next_short_topic(store)
    store.record_topic(topic_id=first.topic_id, series=first.series, title="T1", format_type="short")
    second = pick_next_short_topic(store)
    assert second.topic_id != first.topic_id
    assert second.series != first.series


def test_documentary_picker_skips_archived_places(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    first_doc = DOCUMENTARY_TOPICS[0]
    store.record_documentary(first_doc.place, "vid123", "Title", 420.0)
    picked = pick_next_documentary_topic(store)
    assert picked.slug != first_doc.slug


def test_short_script_validation_rejects_banned_openings_and_short_scripts() -> None:
    topic = SHORT_TOPICS[0]
    bad_opening = ShortScript(
        title="Dead Trees In Forests #Shorts...",
        thumbnail_hook="watch out above",
        pill_label="camping safety",
        description_hook="Never ignore dead branches above your camp. Would you check first?",
        pinned_comment="Have you ever spotted a widowmaker at camp?",
        tags=["#camping", "shorts", "forest safety", "hiking"],
        beats=[
            ShortBeat(narration="Welcome to this video about dead standing trees in the forest.", visual_query="pine forest"),
            ShortBeat(narration="Experienced campers always scan the canopy directly above their tent pad.", visual_query="camping tent forest"),
            ShortBeat(narration="Dead pines lose root strength while looking totally solid from eye level.", visual_query="dead tree trunk"),
            ShortBeat(narration="Even a calm midnight gust can snap a heavy limb cleanly onto nylon.", visual_query="windy pine trees"),
            ShortBeat(narration="Rangers call these hanging branches widowmakers for a very grim reason.", visual_query="fallen tree woods"),
            ShortBeat(narration="Always inspect a thirty-foot circle overhead before staking a single corner.", visual_query="camper looking up"),
        ],
    )
    with pytest.raises(ValueError, match="banned generic phrasing"):
        validate_and_polish_short_script(bad_opening, topic)

    good = bad_opening.model_copy(
        update={
            "beats": [
                ShortBeat(
                    narration="Never pitch your tent until you stare straight up into the tree canopy.",
                    visual_query="tall pine trees looking up",
                ),
                *bad_opening.beats[1:],
            ]
        }
    )
    polished = validate_and_polish_short_script(good, topic)
    assert "#" not in polished.title
    assert "..." not in polished.title
    assert "shorts" not in polished.tags
    assert polished.thumbnail_hook == "WATCH OUT ABOVE"

    desc = build_short_description(polished, topic, [{"url": "https://en.wikipedia.org/wiki/Widowmaker_(forestry)"}])
    assert "CONTENT_HASH" not in desc
    assert "#Shorts" in desc


def test_documentary_description_includes_clickable_chapters() -> None:
    topic = DOCUMENTARY_TOPICS[0]
    script = DocumentaryScript(
        title="Why Banff's Glacial Lakes Glow Neon Turquoise",
        thumbnail_hook="NEON GLACIER LAKES",
        description_intro="Deep in the Canadian Rockies, rock flour and melting ice create impossible color.",
        pinned_comment="Which Rockies trail is on your bucket list?",
        tags=["banff", "rockies", "canada"],
        chapters=[
            DocumentaryChapter(chapter_title="Into The Rockies", narration="Word " * 170, visual_queries=["banff lake"]),
            DocumentaryChapter(chapter_title="Glacial Rock Flour", narration="Word " * 170, visual_queries=["glacier ice"]),
            DocumentaryChapter(chapter_title="Grizzly Country", narration="Word " * 170, visual_queries=["bear forest"]),
            DocumentaryChapter(chapter_title="Backcountry Reality", narration="Word " * 170, visual_queries=["alpine tent"]),
        ],
    )
    desc = build_documentary_description(
        script,
        topic,
        [(0.0, "Into The Rockies"), (82.4, "Glacial Rock Flour"), (168.0, "Grizzly Country")],
        [{"url": "https://parks.canada.ca/pn-np/ab/banff"}],
        ["Wikimedia Commons: Example Author (CC BY 4.0) https://commons.wikimedia.org/wiki/File:Test.webm"],
    )
    assert "00:00 Into The Rockies" in desc
    assert "01:22 Glacial Rock Flour" in desc
    assert "02:48 Grizzly Country" in desc
    assert "CONTENT_HASH" not in desc


def test_pixabay_query_sanitizer_prevents_http_400() -> None:
    long_dirty = "Sørvágsvatn / Faroe Islands: ultra-dramatic!! cliffs & ocean waves (4K) " * 3
    clean = sanitize_query(long_dirty, max_len=70)
    assert len(clean) <= 70
    assert "&" not in clean and "!" not in clean and "(" not in clean


def test_karaoke_ass_and_srt_generation(tmp_path: Path) -> None:
    assert detect_bold_font_path() is not None, "Expected bold TrueType font installed in Linux environment"
    words = build_word_timings(
        "Never pitch your tent inside a dry desert riverbed.",
        [],
        timeline_offset=0.0,
        clip_duration=3.2,
    )
    beats = [BeatAudio(index=0, text="Never pitch your tent inside a dry desert riverbed.", audio_path=tmp_path / "b.wav", start=0.0, duration=3.2, words=words)]
    bursts = group_words_into_bursts(words, max_words=3, max_chars=19)
    assert len(bursts) >= 2

    ass_path = generate_shorts_karaoke_ass(beats, tmp_path / "short.ass", pill_label="CANYON SAFETY")
    ass_text = ass_path.read_text(encoding="utf-8")
    assert "TopPill" in ass_text
    assert "CANYON SAFETY" in ass_text
    assert r"\c&H003BD4FF&" in ass_text

    doc_ass = generate_documentary_chapters_ass([(0.0, 60.0, 1, "Cold Open")], tmp_path / "doc.ass")
    assert "CHAPTER 01" in doc_ass.read_text(encoding="utf-8")

    cues = beats_to_srt_cues(beats)
    srt_path = write_srt(cues, tmp_path / "en.srt")
    assert "00:00:00,000 -->" in srt_path.read_text(encoding="utf-8")
    assert format_ass_time(65.25) == "0:01:05.25"
    assert format_srt_time(65.25) == "00:01:05,250"


def test_analytics_rebalance_rewards_retention_and_stays_bounded() -> None:
    stats = {
        "CAMP_MISTAKES": {"count": 5, "avg_views": 4500, "avg_avp": 94.0, "net_subs": 12},
        "WILD_STAYS": {"count": 4, "avg_views": 300, "avg_avp": 46.0, "net_subs": 0},
        "GEAR_AND_HACKS": {"count": 1, "avg_views": 99999, "avg_avp": 120.0, "net_subs": 50},  # ignored (<2 videos)
    }
    updated = rebalance_series_weights(stats, DEFAULT_SERIES_WEIGHTS)
    assert abs(sum(updated.values()) - 1.0) < 0.01
    assert updated["CAMP_MISTAKES"] > DEFAULT_SERIES_WEIGHTS["CAMP_MISTAKES"]
    assert updated["WILD_STAYS"] < DEFAULT_SERIES_WEIGHTS["WILD_STAYS"]
    for val in updated.values():
        assert 0.08 <= val <= 0.45


def test_youtube_channel_lock_and_http_308_fix(tmp_path: Path) -> None:
    class FakeChannels:
        def __init__(self, cid: str) -> None:
            self.cid = cid

        def list(self, **kwargs):
            return SimpleNamespace(
                execute=lambda: {
                    "items": [
                        {
                            "id": self.cid,
                            "snippet": {"title": "CampUncharted"},
                            "statistics": {"subscriberCount": "39", "viewCount": "31910", "videoCount": "241"},
                            "contentDetails": {"relatedPlaylists": {"uploads": "UUtest"}},
                        }
                    ]
                }
            )

    ok_yt = SimpleNamespace(channels=lambda: FakeChannels(EXPECTED_CHANNEL_ID))
    info = verify_channel_ownership(ok_yt, EXPECTED_CHANNEL_ID)
    assert info["channel_id"] == EXPECTED_CHANNEL_ID

    wrong_yt = SimpleNamespace(channels=lambda: FakeChannels("UC_WRONG_CHANNEL"))
    with pytest.raises(RuntimeError, match="CHANNEL SAFETY LOCK"):
        verify_channel_ownership(wrong_yt, EXPECTED_CHANNEL_ID)

    # Verify HTTP 308 fix is applied so resumable uploads never hang or leave stuck private shells
    dummy_creds = Credentials(token="dummy")
    authed_http = _build_resumable_http(dummy_creds)
    assert 308 not in authed_http.http.redirect_codes

    # Verify peak UTC slot scheduling never double-books
    store = StateStore(tmp_path / "sched")
    now = datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc)
    slot1 = reserve_publish_slot(store, format_type="short", publish_now=False, now_utc=now)
    assert slot1 == "2026-11-20T13:15:00Z"
    record_reserved_slot(store, publish_at=slot1, format_type="short", video_id="v1", title="First")
    slot2 = reserve_publish_slot(store, format_type="short", publish_now=False, now_utc=now)
    assert slot2 == "2026-11-20T18:15:00Z"
    assert reserve_publish_slot(store, format_type="short", publish_now=True, now_utc=now) is None


def test_gemini_model_priority_and_404_fallback() -> None:
    from campuncharted.content_engine import (
        TranslationBatch,
        _generate_structured_with_fallback,
        list_candidate_gemini_models,
    )

    fake_models = [
        SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.5-flash-lite", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-3.5-flash", supported_actions=["generateContent"]),
    ]
    called: list[str] = []

    class FakeModelsApi:
        def list(self):
            return fake_models

        def generate_content(self, *, model: str, contents: str, config: dict):
            called.append(model)
            if model == "gemini-3.8-flash":
                raise RuntimeError("404 NOT_FOUND: temporary unavailable")
            return SimpleNamespace(
                text='{"localized_title":"Test","localized_description":"Desc","translated_lines":["Line 1"]}'
            )

    fake_client = SimpleNamespace(models=FakeModelsApi())
    ordered = list_candidate_gemini_models(fake_client)
    assert ordered[0] == "gemini-3.8-flash"
    assert ordered[1] == "gemini-3.5-flash"

    used_model, parsed = _generate_structured_with_fallback(
        fake_client,
        prompt="test",
        schema=TranslationBatch,
        temperature=0.3,
    )
    assert called == ["gemini-3.8-flash", "gemini-3.5-flash"]
    assert used_model == "gemini-3.5-flash"
    assert parsed.localized_title == "Test"

