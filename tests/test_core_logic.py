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
            if model == "gemini-3.5-flash":
                raise RuntimeError("504 DEADLINE_EXCEEDED: temporary spike")
            return SimpleNamespace(
                text='{"localized_title":"Test","localized_description":"Desc","translated_lines":["Line 1"]}'
            )

    fake_client = SimpleNamespace(models=FakeModelsApi())
    ordered = list_candidate_gemini_models(fake_client)
    assert ordered[0] == "gemini-3.5-flash"
    assert ordered[1] == "gemini-3.8-flash"

    used_model, parsed = _generate_structured_with_fallback(
        fake_client,
        prompt="test",
        schema=TranslationBatch,
        temperature=0.3,
    )
    assert called == ["gemini-3.5-flash", "gemini-3.8-flash"]
    assert used_model == "gemini-3.8-flash"
    assert parsed.localized_title == "Test"


def test_decode_json_secret_handles_all_playground_paste_variants() -> None:
    from campuncharted.youtube_engine import _decode_json_secret

    sample_rt = "1//04FAKE_REFRESH_TOKEN_FOR_UNIT_TEST_1234567890"
    # 1. Full Playground HTTP dump
    http_dump = f'HTTP/1.1 200 OK\nContent-Type: application/json\n\n{{"refresh_token": "{sample_rt}", "access_token": "ya29.test"}}'
    assert _decode_json_secret(http_dump)["refresh_token"] == sample_rt

    # 2. Partial JSON missing outer braces
    partial = f'"refresh_token": "{sample_rt}",'
    assert _decode_json_secret(partial)["refresh_token"] == sample_rt

    # 3. Bare refresh token
    assert _decode_json_secret(sample_rt)["refresh_token"] == sample_rt


def test_trend_radar_outlier_scoring_and_topic_blending(tmp_path: Path) -> None:
    from campuncharted.state_store import atomic_write_json
    from campuncharted.trend_radar import (
        compute_outlier_score,
        parse_iso8601_duration,
        select_daily_trend_queries,
    )

    assert parse_iso8601_duration("PT48S") == 48
    assert parse_iso8601_duration("PT1M15S") == 75
    assert len(select_daily_trend_queries(day_of_year=10, count=2)) == 2

    small_viral = compute_outlier_score(views=250_000, likes=12_000, comments=450, channel_subs=8_000, age_days=3.0)
    big_average = compute_outlier_score(views=25_000, likes=500, comments=10, channel_subs=1_500_000, age_days=14.0)
    assert small_viral["viral_score"] > big_average["viral_score"] * 2.0

    # Verify pick_next_short_topic alternates trending_topics.json ideas with evergreen catalog
    store = StateStore(tmp_path / "trend_state")
    atomic_write_json(
        store.state_dir / "trending_topics.json",
        {
            "topics": [
                {
                    "topic_id": "trend_winter_tent_candle_myth",
                    "series": "CAMP_MISTAKES",
                    "subject": "Does a Single Candle Really Heat a Freezing Winter Tent?",
                    "angle_hook": "Why the viral candle-in-a-tent winter hack can create a silent hazard",
                    "wiki_titles": ["Winter_camping"],
                    "visual_queries": ["winter snow tent glowing night"],
                    "mood": "CAMPFIRE_NIGHT",
                }
            ]
        },
    )
    picked_1 = pick_next_short_topic(store)
    assert picked_1.topic_id == "trend_winter_tent_candle_myth"
    store.record_topic(topic_id=picked_1.topic_id, series=picked_1.series, title="T1", format_type="short")
    picked_2 = pick_next_short_topic(store)
    assert not picked_2.topic_id.startswith("trend_")


def test_comment_engine_filters_spam_owner_and_replied_threads() -> None:
    from campuncharted.comment_engine import (
        filter_unreplied_viewer_threads,
        is_obvious_spam,
    )

    assert is_obvious_spam("Check out my channel sub4sub https://spam.example")
    assert is_obvious_spam("DM me on WhatsApp for crypto signals")
    assert not is_obvious_spam("Great tip! Never knew pine resin burned in rain 🔥")
    assert not is_obvious_spam("Harika video, kampa giderken mutlaka denerim!")

    threads = [
        # 1. Own channel starter comment -> MUST skip
        {
            "id": "th_own",
            "snippet": {
                "videoId": "vid1",
                "topLevelComment": {
                    "id": "c_own",
                    "snippet": {
                        "authorChannelId": {"value": EXPECTED_CHANNEL_ID},
                        "authorDisplayName": "CampUncharted",
                        "textOriginal": "Have you ever camped in rain?",
                    },
                },
            },
        },
        # 2. Viewer comment already replied by owner -> MUST skip
        {
            "id": "th_done",
            "snippet": {
                "videoId": "vid1",
                "topLevelComment": {
                    "id": "c_done",
                    "snippet": {
                        "authorChannelId": {"value": "UC_VIEWER_1"},
                        "authorDisplayName": "Alex",
                        "textOriginal": "Love this!",
                    },
                },
            },
            "replies": {
                "comments": [
                    {"snippet": {"authorChannelId": {"value": EXPECTED_CHANNEL_ID}, "textOriginal": "Thanks Alex!"}}
                ]
            },
        },
        # 3. Fresh unreplied viewer comment -> MUST include
        {
            "id": "th_new",
            "snippet": {
                "videoId": "vid2",
                "topLevelComment": {
                    "id": "c_new",
                    "snippet": {
                        "authorChannelId": {"value": "UC_VIEWER_2"},
                        "authorDisplayName": "Mert",
                        "textOriginal": "Çam reçinesi gerçekten ıslakken bile yanıyor mu?",
                    },
                },
            },
        },
    ]
    pending = filter_unreplied_viewer_threads(
        threads,
        owner_channel_id=EXPECTED_CHANNEL_ID,
        processed_thread_ids={"th_old"},
    )
    assert [p["thread_id"] for p in pending] == ["th_new"]
    assert pending[0]["video_id"] == "vid2"


def test_comment_engine_handles_viewer_replies_on_owner_starter_comment_and_paginates(tmp_path: Path) -> None:
    from campuncharted.comment_engine import (
        filter_unreplied_viewer_threads,
        run_comment_engagement,
    )

    starter_thread_with_viewer_reply = {
        "id": "th_starter",
        "snippet": {
            "videoId": "vid_short_1",
            "topLevelComment": {
                "id": "c_starter",
                "snippet": {
                    "authorChannelId": {"value": EXPECTED_CHANNEL_ID},
                    "authorDisplayName": "CampUncharted",
                    "textOriginal": "Have you ever been caught in soaked clothing on a freezing trail?",
                },
            },
        },
        "replies": {
            "comments": [
                {
                    "id": "rep_old_viewer",
                    "snippet": {
                        "authorChannelId": {"value": "UC_V1"},
                        "authorDisplayName": "OldViewer",
                        "textOriginal": "Yes once in 2022!",
                        "publishedAt": "2026-11-23T10:00:00Z",
                    },
                },
                {
                    "id": "rep_owner_ack",
                    "snippet": {
                        "authorChannelId": {"value": EXPECTED_CHANNEL_ID},
                        "authorDisplayName": "CampUncharted",
                        "textOriginal": "Glad you made it out warm!",
                        "publishedAt": "2026-11-23T11:00:00Z",
                    },
                },
                {
                    "id": "rep_new_viewer",
                    "snippet": {
                        "authorChannelId": {"value": "UC_V2"},
                        "authorDisplayName": "Deniz",
                        "textOriginal": "Kaçkar'da başıma geldi, merino yün hayat kurtarıyor!",
                        "publishedAt": "2026-11-23T12:00:00Z",
                    },
                },
            ]
        },
    }
    pending = filter_unreplied_viewer_threads(
        [starter_thread_with_viewer_reply],
        owner_channel_id=EXPECTED_CHANNEL_ID,
        processed_thread_ids=set(),
    )
    assert len(pending) == 1
    assert pending[0]["thread_id"] == "rep_new_viewer"
    assert pending[0]["parent_id"] == "c_starter"
    assert pending[0]["video_id"] == "vid_short_1"

    # Verify pagination reaches page 2 when page 1 only has owner starter comments
    page_calls: list[str | None] = []

    class FakeCommentThreads:
        def list(self, **kwargs):
            tok = kwargs.get("pageToken")
            page_calls.append(tok)
            if tok is None:
                return SimpleNamespace(
                    execute=lambda: {
                        "items": [
                            {
                                "id": "th_own_only",
                                "snippet": {
                                    "videoId": "v1",
                                    "topLevelComment": {
                                        "id": "c1",
                                        "snippet": {
                                            "authorChannelId": {"value": EXPECTED_CHANNEL_ID},
                                            "textOriginal": "Question?",
                                        },
                                    },
                                },
                            }
                        ],
                        "nextPageToken": "PAGE2",
                    }
                )
            return SimpleNamespace(execute=lambda: {"items": [starter_thread_with_viewer_reply]})

    class FakeVideos:
        def list(self, **kwargs):
            return SimpleNamespace(execute=lambda: {"items": [{"id": "vid_short_1", "snippet": {"title": "Cotton Rule"}}]})

    fake_yt = SimpleNamespace(commentThreads=lambda: FakeCommentThreads(), videos=lambda: FakeVideos())
    store = StateStore(tmp_path / "comment_state")
    stats = run_comment_engagement(fake_yt, store, api_key="", max_replies=8, dry_run=True)
    assert page_calls == [None, "PAGE2"]
    assert stats["scanned"] == 2
    assert stats["new_viewer_comments"] == 1


def test_subtitle_queue_decouples_comments_prevents_starvation_and_preserves_state(tmp_path: Path) -> None:
    from campuncharted.config import ApiCredentials
    from campuncharted.youtube_engine import (
        enqueue_video_for_post_processing,
        process_subtitle_queue,
    )

    store = StateStore(tmp_path / "sub_state")
    cues = [(0.0, 2.0, "Never camp on a dry canyon floor.")]
    enqueue_video_for_post_processing(
        store,
        video_id="vid_old",
        title="Old Video",
        description="Old desc",
        cues=cues,
        pinned_comment="Would you camp here?",
        comment_posted=False,
        publish_at="2026-11-23T13:15:00Z",
    )
    enqueue_video_for_post_processing(
        store,
        video_id="vid_new",
        title="New Video",
        description="New desc",
        cues=cues,
        pinned_comment="Have you tried this?",
        comment_posted=False,
        publish_at="2026-11-24T13:15:00Z",
    )

    # Re-enqueuing an existing video MUST NOT wipe done_langs or comment_posted
    q = store.get_subtitle_queue()
    q["videos"]["vid_old"]["done_langs"] = ["en"]
    q["videos"]["vid_old"]["localized_langs"] = ["es"]
    store.save_subtitle_queue(q)
    enqueue_video_for_post_processing(
        store,
        video_id="vid_old",
        title="Old Video",
        description="Old desc",
        cues=cues,
        pinned_comment="Would you camp here?",
        comment_posted=False,
    )
    q_after = store.get_subtitle_queue()
    assert q_after["videos"]["vid_old"]["done_langs"] == ["en"]
    assert q_after["videos"]["vid_old"]["localized_langs"] == ["es"]

    # Even when today's caption quota is ALREADY FULL (1800/1800), due delayed comment MUST STILL BE POSTED!
    q_after["quota"] = {"2026-11-24": 1800}
    store.save_subtitle_queue(q_after)

    posted_comments: list[str] = []
    uploaded_captions: list[tuple[str, str]] = []

    class FakeComments:
        def insert(self, **kwargs):
            vid = kwargs["body"]["snippet"]["videoId"]
            posted_comments.append(vid)
            return SimpleNamespace(execute=lambda: {"id": "c_posted"})

    class FakeCaptions:
        def insert(self, **kwargs):
            snip = kwargs["body"]["snippet"]
            uploaded_captions.append((snip["videoId"], snip["language"]))
            return SimpleNamespace(execute=lambda: {"id": "cap_id"})

    fake_yt = SimpleNamespace(commentThreads=lambda: FakeComments(), captions=lambda: FakeCaptions())
    now = datetime(2026, 11, 24, 5, 0, 0, tzinfo=timezone.utc)
    stats_full = process_subtitle_queue(
        fake_yt,
        store,
        ApiCredentials(gemini_api_key=""),
        tmp_path / "work1",
        now_utc=now,
    )
    assert posted_comments == ["vid_old"]  # vid_new is scheduled for 13:15 UTC so not due yet
    assert stats_full["comments_posted"] == 1
    assert stats_full["captions_uploaded"] == 0

    # On a fresh day with max_captions=1, vid_new gets English caption without burning all daily budget
    next_day = datetime(2026, 11, 25, 8, 0, 0, tzinfo=timezone.utc)
    stats_fresh = process_subtitle_queue(
        fake_yt,
        store,
        ApiCredentials(gemini_api_key=""),
        tmp_path / "work2",
        max_captions=1,
        now_utc=next_day,
    )
    assert uploaded_captions == [("vid_new", "en")]
    assert stats_fresh["captions_uploaded"] == 1
    assert stats_fresh["comments_posted"] == 1  # vid_new is now past 2026-11-24T13:15Z!
    assert stats_fresh["quota_used_today"] == 400


def test_analytics_preserves_live_trends_and_deduplicates_daily_snapshots(tmp_path: Path) -> None:
    from campuncharted.analytics_engine import run_analytics_optimization

    store = StateStore(tmp_path / "ana_state")
    store.save_channel_insights(
        {
            "series_weights": dict(DEFAULT_SERIES_WEIGHTS),
            "live_youtube_trends": [
                "Visual contrast between fatal mistakes and survival hacks drives saves.",
                "Short fire-starting bushcraft physics hooks viewers in 2 seconds.",
            ],
        }
    )

    class FakeReports:
        def query(self, **kwargs):
            if "dimensions" in kwargs:
                return SimpleNamespace(execute=lambda: {"columnHeaders": [], "rows": []})
            return SimpleNamespace(
                execute=lambda: {
                    "columnHeaders": [
                        {"name": "views"},
                        {"name": "estimatedMinutesWatched"},
                        {"name": "averageViewDuration"},
                        {"name": "averageViewPercentage"},
                        {"name": "subscribersGained"},
                        {"name": "subscribersLost"},
                        {"name": "likes"},
                        {"name": "comments"},
                    ],
                    "rows": [[2500, 380.0, 24.0, 38.5, 10, 2, 40, 3]],
                }
            )

    fake_yt = SimpleNamespace(videos=lambda: SimpleNamespace(list=lambda **kw: SimpleNamespace(execute=lambda: {"items": []})))
    fake_ana = SimpleNamespace(reports=lambda: FakeReports())
    ch_info = {"subscribers": 30, "views": 35000, "video_count": 245}

    out1 = run_analytics_optimization(fake_yt, fake_ana, store, ch_info)
    assert len(out1["live_youtube_trends"]) == 2
    assert "Live YouTube trends:" in out1["summary_guidance"]

    # Running a second time on the same UTC day must NOT create duplicate date entries
    run_analytics_optimization(fake_yt, fake_ana, store, ch_info)
    hist = store.get_analytics_history()["snapshots"]
    assert len(hist) == 1


def test_trend_radar_and_catalog_block_duplicate_concepts_and_locations(tmp_path: Path) -> None:
    import json
    from campuncharted.state_store import atomic_write_json
    from campuncharted.trend_radar import synthesize_trending_topics

    store = StateStore(tmp_path / "dedup_trends")
    store.record_topic(
        topic_id="trend_pine_resin_tinder_survival_hack",
        series="GEAR_AND_HACKS",
        title="Why Wise Campers Collect Pine Resin in Rain",
        video_id="tfYyd6e93jM",
        format_type="short",
    )
    # Record one evergreen short afterwards so pick_next_short_topic prefers unused_trending
    store.record_topic(
        topic_id="eurasia_goygol_azerbaijan",
        series="TURKEY_AND_EURASIA_HIDDEN",
        title="The Mountain Lake Born From a Catastrophic Earthquake",
        video_id="eM9KEwi1S1M",
        format_type="short",
    )
    atomic_write_json(
        store.state_dir / "trending_topics.json",
        {
            "topics": [
                {
                    "topic_id": "trend_pine_resin_fatwood_fire",
                    "series": "GEAR_AND_HACKS",
                    "subject": "Harvesting pine resin fatwood for wet fire starting",
                    "angle_hook": "Natural waterproof fire starter",
                    "location_name": "",
                },
                {
                    "topic_id": "trend_kackar_1",
                    "series": "TURKEY_AND_EURASIA_HIDDEN",
                    "subject": "Glacial lakes of Kaçkar Mountains",
                    "angle_hook": "Alpine tarns",
                    "location_name": "Kaçkar Mountains National Park, Turkey",
                },
                {
                    "topic_id": "trend_kackar_2",
                    "series": "TURKEY_AND_EURASIA_HIDDEN",
                    "subject": "Hidden turquoise lakes in Kaçkar Mountains",
                    "angle_hook": "Alpine tarns 2",
                    "location_name": "Kaçkar Mountains National Park, Turkey",
                },
                {
                    "topic_id": "trend_unique_compass_declination",
                    "series": "CAMP_MISTAKES",
                    "subject": "Why ignoring magnetic declination leads hikers miles off course",
                    "angle_hook": "Your compass does not point to true north",
                    "location_name": "",
                },
            ]
        },
    )

    # Pine resin duplicate must NEVER be selected even when trending pool is preferred
    picked = pick_next_short_topic(store)
    assert picked.topic_id != "trend_pine_resin_fatwood_fire"
    assert picked.topic_id in {"trend_kackar_1", "trend_unique_compass_declination"}

    # Verify synthesize_trending_topics injects covered topics into prompt and prunes duplicates
    prompts_seen: list[str] = []

    class FakeTrendModels:
        def list(self):
            return [SimpleNamespace(name="models/gemini-3.5-flash", supported_actions=["generateContent"])]

        def generate_content(self, *, model: str, contents: str, config: dict):
            prompts_seen.append(contents)
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "trend_patterns": ["Practical navigation rules earn high saves."],
                        "ideas": [
                            {
                                "topic_id": "trend_pine_resin_duplicate_2",
                                "series": "GEAR_AND_HACKS",
                                "subject": "Collecting pine resin for wet campfire ignition",
                                "angle_hook": "Pine resin burns in rain",
                                "wiki_titles": ["Pine_resin"],
                                "visual_queries": ["pine tree resin"],
                                "location_name": "",
                                "mood": "FOREST_RAIN",
                                "inspired_by_pattern": "Bushcraft fire hack",
                            },
                            {
                                "topic_id": "trend_kackar_duplicate_3",
                                "series": "TURKEY_AND_EURASIA_HIDDEN",
                                "subject": "Backcountry camping in Kaçkar Mountains",
                                "angle_hook": "Alpine ridge",
                                "wiki_titles": ["Kaçkar_Mountains"],
                                "visual_queries": ["kackar mountains turkey"],
                                "location_name": "Kaçkar Mountains National Park, Turkey",
                                "mood": "ALPINE_WIND",
                                "inspired_by_pattern": "Hidden Eurasia spot",
                            },
                            {
                                "topic_id": "trend_altitude_boil_cooking_myth",
                                "series": "EXTREME_WEATHER_NATURE",
                                "subject": "Why boiling water at high altitude leaves backcountry meals undercooked",
                                "angle_hook": "Lower air pressure changes boiling physics",
                                "wiki_titles": ["High-altitude_cooking"],
                                "visual_queries": ["camping stove mountain snow"],
                                "location_name": "",
                                "mood": "ALPINE_WIND",
                                "inspired_by_pattern": "Counter-intuitive mountain physics",
                            },
                            {
                                "topic_id": "trend_paracord_wet_stretch_guylines",
                                "series": "GEAR_AND_HACKS",
                                "subject": "Why nylon paracord sags in overnight rain while dyneema guylines stay taut",
                                "angle_hook": "Hydrophilic fibers silently collapse taut tarps",
                                "wiki_titles": ["Parachute_cord"],
                                "visual_queries": ["rain tarp guyline forest"],
                                "location_name": "",
                                "mood": "FOREST_RAIN",
                                "inspired_by_pattern": "Practical shelter rigging physics",
                            },
                        ],
                    }
                )
            )

    import campuncharted.trend_radar as trend_radar_mod

    orig_client = trend_radar_mod.genai.Client
    trend_radar_mod.genai.Client = lambda **kw: SimpleNamespace(models=FakeTrendModels())
    try:
        res = synthesize_trending_topics(
            [
                {
                    "title": "Camping hack #shorts",
                    "views": 50000,
                    "age_days": 3.0,
                    "views_per_day": 16666.0,
                    "outlier_ratio": 12.0,
                    "duration_sec": 35,
                }
            ],
            store,
            api_key="dummy",
        )
    finally:
        trend_radar_mod.genai.Client = orig_client

    assert prompts_seen and "Pine Resin" in prompts_seen[0]
    assert res["added"] == 1  # Only genuinely new paracord idea added; pine resin, Kackar & catalog boiling duplicates rejected!
    saved = json.loads((store.state_dir / "trending_topics.json").read_text(encoding="utf-8"))
    saved_ids = [t["topic_id"] for t in saved["topics"]]
    assert "trend_paracord_wet_stretch_guylines" in saved_ids
    assert "trend_altitude_boil_cooking_myth" not in saved_ids
    assert "trend_pine_resin_fatwood_fire" not in saved_ids
    assert "trend_pine_resin_duplicate_2" not in saved_ids
    assert "trend_kackar_2" not in saved_ids
    assert "trend_kackar_duplicate_3" not in saved_ids


def test_wikimedia_asset_key_is_deterministic_sha256() -> None:
    import hashlib
    from campuncharted.media_engine import search_wikimedia_commons

    title = "File:Banff_Moraine_Lake_Sunrise.webm"
    expected_key = f"wmc_{hashlib.sha256(title.encode('utf-8')).hexdigest()[:12]}"

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "query": {
                    "pages": {
                        "1": {
                            "title": title,
                            "imageinfo": [
                                {
                                    "mime": "video/webm",
                                    "width": 1920,
                                    "height": 1080,
                                    "size": 5_000_000,
                                    "duration": 14.0,
                                    "url": "https://upload.wikimedia.org/test.webm",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/" + title,
                                    "extmetadata": {
                                        "LicenseShortName": {"value": "CC BY 4.0"},
                                        "Artist": {"value": "Jane Doe"},
                                    },
                                }
                            ],
                        }
                    }
                }
            }

    fake_session = SimpleNamespace(get=lambda *a, **kw: FakeResp())
    assets = search_wikimedia_commons("Banff lake", session=fake_session)
    assert len(assets) == 1
    assert assets[0].asset_key == expected_key


def test_doc_validation_chunked_translation_and_invalid_json_fallback() -> None:
    import json
    from campuncharted.content_engine import (
        TranslationBatch,
        _generate_structured_with_fallback,
        translate_subtitles_and_metadata,
        validate_and_polish_documentary_script,
    )

    topic = DOCUMENTARY_TOPICS[0]
    raw_doc = DocumentaryScript(
        title="Banff Hidden Glacial Valleys #Documentary...",
        thumbnail_hook="NEON GLACIER LAKES",
        description_intro="Deep in the Canadian Rockies.",
        pinned_comment="Have you hiked Banff?",
        tags=["#banff", "<script>", "rockies", "banff", "a" * 40],
        chapters=[
            DocumentaryChapter(chapter_title="Chapter 1: Into The Ice", narration="Chapter 1: " + ("glacial " * 130), visual_queries=[]),
            DocumentaryChapter(chapter_title="Chapter 2: Rock Flour", narration="mineral " * 130, visual_queries=["banff lake"]),
            DocumentaryChapter(chapter_title="Chapter 3: Grizzly Trails", narration="wildlife " * 130, visual_queries=["bear forest"]),
            DocumentaryChapter(chapter_title="Chapter 4: Alpine Dusk", narration="mountain " * 130, visual_queries=["alpine dusk"]),
        ],
    )
    polished = validate_and_polish_documentary_script(raw_doc, topic)
    assert "#" not in polished.title and "..." not in polished.title
    assert all("<" not in t and ">" not in t and "#" not in t and len(t) <= 30 for t in polished.tags)
    assert polished.chapters[0].chapter_title == "Into The Ice"
    assert not polished.chapters[0].narration.startswith("Chapter 1:")
    assert len(polished.chapters[0].visual_queries) >= 1

    # Verify _generate_structured_with_fallback falls back on empty / malformed JSON response
    models_called: list[str] = []

    class FakeFallbackModels:
        def list(self):
            return [
                SimpleNamespace(name="models/gemini-3.5-flash", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"]),
            ]

        def generate_content(self, *, model: str, contents: str, config: dict):
            models_called.append(model)
            if model == "gemini-3.5-flash":
                return SimpleNamespace(text="")  # Empty response -> must fall back, not crash!
            return SimpleNamespace(
                text=json.dumps({"localized_title": "Ok", "localized_description": "Desc", "translated_lines": ["A"]})
            )

    used, parsed = _generate_structured_with_fallback(
        SimpleNamespace(models=FakeFallbackModels()),
        prompt="hi",
        schema=TranslationBatch,
        temperature=0.2,
    )
    assert models_called == ["gemini-3.5-flash", "gemini-3.8-flash"]
    assert used == "gemini-3.8-flash"
    assert parsed.localized_title == "Ok"

    # Verify chunked subtitle translation handles 60 lines in chunks of 25 (25 + 25 + 10)
    chunk_sizes_seen: list[int] = []

    class FakeChunkModels:
        def list(self):
            return [SimpleNamespace(name="models/gemini-3.5-flash", supported_actions=["generateContent"])]

        def generate_content(self, *, model: str, contents: str, config: dict):
            json_blob = contents.split("INPUT JSON:\n", 1)[1].strip()
            data = json.loads(json_blob)
            lines = data["lines"]
            chunk_sizes_seen.append(len(lines))
            if "title" in data:
                out = {
                    "localized_title": "Titulo",
                    "localized_description": "Descripcion",
                    "translated_lines": [f"ES:{x}" for x in lines],
                }
            else:
                out = {"translated_lines": [f"ES:{x}" for x in lines]}
            return SimpleNamespace(text=json.dumps(out))

    res = translate_subtitles_and_metadata(
        title="Doc Title",
        description="Long documentary description with chapters",
        lines=[f"Line {i}" for i in range(60)],
        target_lang="es",
        api_key="dummy",
        chunk_size=25,
        client=SimpleNamespace(models=FakeChunkModels()),
    )
    assert chunk_sizes_seen == [25, 25, 10]
    assert len(res.translated_lines) == 60
    assert res.translated_lines[0] == "ES:Line 0"
    assert res.translated_lines[-1] == "ES:Line 59"


def test_ass_hook_banner_hex_and_word_timing_bounds(tmp_path: Path) -> None:
    # Verify 8-digit ASS BackColour hex literal in HookBanner
    words = build_word_timings(
        "Stay dry out there.",
        [
            {"type": "WordBoundary", "offset_sec": 0.1, "duration_sec": 0.3, "text": "Stay"},
            # Overshooting offset past clip_duration must clamp cleanly so end >= start
            {"type": "WordBoundary", "offset_sec": 5.0, "duration_sec": 0.2, "text": "dry"},
        ],
        timeline_offset=0.0,
        clip_duration=1.5,
    )
    assert all(w.end >= w.start for w in words)
    beats = [BeatAudio(index=0, text="Stay dry", audio_path=tmp_path / "b.wav", start=0.0, duration=1.5, words=words)]
    ass_path = generate_shorts_karaoke_ass(beats, tmp_path / "hook.ass", hook_banner="STAY DRY")
    ass_text = ass_path.read_text(encoding="utf-8")
    assert "&HA0000000," in ass_text
    assert "&HA000000," not in ass_text


def test_v3_short_script_guardrails_enforce_fast_hook_and_seamless_loop() -> None:
    topic = SHORT_TOPICS[0]

    def _make_script(first_line: str, last_line: str, extra_beats: int = 0) -> ShortScript:
        beats = [
            ShortBeat(narration=first_line, visual_query="winter tent snow"),
            ShortBeat(narration="Your body weight crushes down insulation flat against cold earth.", visual_query="sleeping bag tent"),
            ShortBeat(narration="Frozen soil steals your heat sixty times faster than air.", visual_query="frozen ground frost"),
            ShortBeat(narration="That heat drain triggers dangerous shivering inside an hour.", visual_query="snowy pine forest"),
            ShortBeat(narration="Slide an insulated pad or thick pine boughs underneath first.", visual_query="pine tree branches"),
        ]
        for i in range(extra_beats):
            beats.append(
                ShortBeat(
                    narration=f"Extra rambling filler beat number {i} that adds way too many unnecessary words to stretch the runtime.",
                    visual_query="mountain forest",
                )
            )
        beats.append(ShortBeat(narration=last_line, visual_query="winter campsite morning"))
        return ShortScript(
            title="Why a $500 Sleeping Bag Still Freezes You",
            thumbnail_hook="NEVER SKIP THIS LAYER PLEASE",
            pill_label="WINTER SURVIVAL RULE",
            description_hook="Ground cold beats expensive bags. What pad do you pack?",
            pinned_comment="What R-value pad do you trust below freezing?",
            tags=["winter camping", "sleeping pad", "camping mistakes"],
            beats=beats,
        )

    # 1. Beat 1 over 13 words must be rejected (too slow for 2s hook)
    slow_hook = _make_script(
        "This really expensive winter sleeping bag you just bought online at the store will completely fail you tonight.",
        "Skip that thin pad underneath and...",
    )
    with pytest.raises(ValueError, match="too slow"):
        validate_and_polish_short_script(slow_hook, topic)

    # 2. Swipe-triggering "Would you...?" closing question must be rejected
    bad_closer = _make_script(
        "A five-hundred-dollar sleeping bag can still freeze you.",
        "Would you ever sleep on snow without a pad?",
    )
    with pytest.raises(ValueError, match="closing question"):
        validate_and_polish_short_script(bad_closer, topic)

    # 3. Valid V3 seamless-loop script trims excess beats, caps thumbnail_hook, and activates punch_zoom
    valid_verbose = _make_script(
        "A five-hundred-dollar sleeping bag can still freeze you.",
        "Skip that one thin layer underneath and...",
        extra_beats=3,
    )
    polished = validate_and_polish_short_script(valid_verbose, topic)
    assert sum(len(b.narration.split()) for b in polished.beats) <= 88
    assert polished.beats[0].punch_zoom is True
    assert polished.beats[-2].punch_zoom is True
    assert len(polished.thumbnail_hook) <= 18
    assert polished.beats[-1].narration.startswith("Skip that one thin layer")


def test_v3_karaoke_ass_alert_highlight_and_clean_hook_banner(tmp_path: Path) -> None:
    line = "Never pay 500 dollars for a deadly trap."
    words = build_word_timings(line, [], timeline_offset=0.0, clip_duration=2.4)
    beats = [BeatAudio(index=0, text=line, audio_path=tmp_path / "b.wav", start=0.0, duration=2.4, words=words)]
    ass_path = generate_shorts_karaoke_ass(
        beats,
        tmp_path / "viral.ass",
        pill_label="WINTER TRAP",
        hook_banner="NEVER DO THIS",
    )
    ass_text = ass_path.read_text(encoding="utf-8")
    # HookBanner present without competing TopPill dialogue at t=0
    assert "Dialogue: 2,0:00:00.00,0:00:01.35,HookBanner" in ass_text
    assert "Dialogue: 1,0:00:00.00" not in ass_text
    # Alert words (NEVER / 500 / DEADLY / TRAP) pop in coral-red, normal words in viral gold
    assert r"\c&H003B4DFF&" in ass_text
    assert r"\c&H003BD4FF&" in ass_text


def test_v3_topic_picker_focuses_on_core_viral_series_unless_overridden(tmp_path: Path) -> None:
    from campuncharted.config import SHORTS_ACTIVE_VIRAL_SERIES

    store = StateStore(tmp_path / "viral_series_state")
    for i in range(6):
        picked = pick_next_short_topic(store)
        assert picked.series in SHORTS_ACTIVE_VIRAL_SERIES
        store.record_topic(topic_id=picked.topic_id, series=picked.series, title=f"Short {i}", format_type="short")

    # Explicit override still allows WILD_STAYS when manually requested
    manual = pick_next_short_topic(store, series_override="WILD_STAYS")
    assert manual.series == "WILD_STAYS"


def test_v3_fetch_beat_synced_clips_matches_every_beat_and_skips_dark_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import campuncharted.media_engine as media_mod
    from campuncharted.config import ApiCredentials
    from campuncharted.media_engine import MediaAsset, fetch_beat_synced_clips

    store = StateStore(tmp_path / "media_state")
    queries_searched: list[str] = []

    def fake_pexels(query: str, *, orientation: str, api_key: str, store: StateStore, per_page: int = 10, session=None):
        queries_searched.append(query)
        slug = query.replace(" ", "_")
        return [
            MediaAsset(
                asset_key=f"pex_{slug}_dark",
                provider="Pexels",
                asset_id=f"{slug}_dark",
                download_url=f"https://example.com/{slug}_dark.mp4",
                page_url="",
                title=slug,
                author="A",
                license_name="Pexels",
                width=1080,
                height=1920,
                duration=12.0,
            ),
            MediaAsset(
                asset_key=f"pex_{slug}_bright1",
                provider="Pexels",
                asset_id=f"{slug}_bright1",
                download_url=f"https://example.com/{slug}_bright1.mp4",
                page_url="",
                title=slug,
                author="B",
                license_name="Pexels",
                width=1080,
                height=1920,
                duration=14.0,
            ),
            MediaAsset(
                asset_key=f"pex_{slug}_bright2",
                provider="Pexels",
                asset_id=f"{slug}_bright2",
                download_url=f"https://example.com/{slug}_bright2.mp4",
                page_url="",
                title=slug,
                author="C",
                license_name="Pexels",
                width=1080,
                height=1920,
                duration=15.0,
            ),
        ]

    def fake_download(asset: MediaAsset, dest_path: Path, *, session=None) -> bool:
        dest_path.write_bytes(b"fake-mp4-bytes")
        asset.local_path = dest_path
        return True

    def fake_probe(path: Path, duration: float, probe_dir: Path, tag: str):
        if "_dark" in path.name:
            return (12.0, 20.0)  # Dark clip must be skipped!
        return (125.0, 48.0)  # Bright, high-contrast clip passes

    monkeypatch.setattr(media_mod, "search_pexels", fake_pexels)
    monkeypatch.setattr(media_mod, "search_pixabay", lambda *a, **kw: [])
    monkeypatch.setattr(media_mod, "download_asset", fake_download)

    beat_queries = ["winter tent snow", "sleeping bag floor", "frozen ground ice", "pine branches forest"]
    matched = fetch_beat_synced_clips(
        beat_queries,
        orientation="portrait",
        work_dir=tmp_path / "work",
        creds=ApiCredentials(pexels_api_key="dummy"),
        store=store,
        clips_per_beat=1,
        hook_clips=2,
        brightness_probe=fake_probe,
    )

    assert len(matched) == 4
    # Every single beat query was searched (no stopping after Beat 2!)
    assert queries_searched == beat_queries
    # Beat 0 skipped the dark clip and got 2 bright clips from Beat 0's own query
    assert len(matched[0]) == 2
    assert all("winter_tent_snow_bright" in c.asset_key for c in matched[0])
    assert all(c.beat_index == 0 for c in matched[0])
    # Beat 3 got a bright clip matching Beat 3's own query
    assert len(matched[3]) == 1
    assert "pine_branches_forest_bright" in matched[3][0].asset_key
    assert matched[3][0].beat_index == 3


def test_v3_subtitle_queue_reserves_daily_quota_for_english_cc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import campuncharted.youtube_engine as yt_mod
    from campuncharted.config import ApiCredentials
    from campuncharted.content_engine import TranslationBatch
    from campuncharted.youtube_engine import enqueue_video_for_post_processing, process_subtitle_queue

    store = StateStore(tmp_path / "reserve_state")
    cues = [(0.0, 2.0, "Never camp under dead trees.")]
    enqueue_video_for_post_processing(
        store,
        video_id="vid_existing",
        title="Existing Short",
        description="Desc",
        cues=cues,
        comment_posted=True,
    )
    # Mark 'en' already done so backfill tries non-English languages (es, tr, de, fr)
    q = store.get_subtitle_queue()
    q["videos"]["vid_existing"]["done_langs"] = ["en"]
    q["videos"]["vid_existing"]["localized_langs"] = ["es", "tr", "de", "fr"]
    store.save_subtitle_queue(q)

    uploaded: list[tuple[str, str]] = []

    class FakeCaptions:
        def insert(self, **kwargs):
            snip = kwargs["body"]["snippet"]
            uploaded.append((snip["videoId"], snip["language"]))
            return SimpleNamespace(execute=lambda: {"id": "cap"})

    fake_yt = SimpleNamespace(captions=lambda: FakeCaptions())
    monkeypatch.setattr(
        yt_mod,
        "translate_subtitles_and_metadata",
        lambda **kw: TranslationBatch(localized_title="T", localized_description="D", translated_lines=["L"]),
    )

    now = datetime(2026, 11, 25, 10, 0, 0, tzinfo=timezone.utc)
    # Daily budget 1800 with 450 EN reserve -> non-English backfill caps at 1350 (max 3x400 = 1200 units)
    stats_backfill = process_subtitle_queue(
        fake_yt,
        store,
        ApiCredentials(gemini_api_key="dummy"),
        tmp_path / "work_bf",
        daily_budget=1800,
        now_utc=now,
    )
    assert stats_backfill["quota_used_today"] == 1200
    assert len(uploaded) == 3  # Left 600 units unspent for fresh uploads' English CC!

    # Later the same day, a brand-new Short is uploaded -> English CC MUST succeed using the reserved quota!
    enqueue_video_for_post_processing(
        store,
        video_id="vid_fresh_today",
        title="Fresh Short",
        description="Desc",
        cues=cues,
        comment_posted=True,
    )
    stats_fresh = process_subtitle_queue(
        fake_yt,
        store,
        ApiCredentials(gemini_api_key=""),
        tmp_path / "work_fresh",
        daily_budget=1800,
        max_captions=1,
        now_utc=now,
    )
    assert ("vid_fresh_today", "en") in uploaded
    assert stats_fresh["captions_uploaded"] == 1
    assert stats_fresh["quota_used_today"] == 1600
