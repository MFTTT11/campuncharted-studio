"""Command-line interface and orchestration pipelines for CampUncharted Studio v2."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analytics_engine import run_analytics_optimization
from .audio_engine import (
    generate_nature_soundscape,
    mix_narration_and_ambience,
    synthesize_narration_beats,
)
from .config import (
    ApiCredentials,
    CATEGORY_TRAVEL_EVENTS,
    DEFAULT_SERIES_WEIGHTS,
    DOC_CHAPTER_GAP_SECONDS,
    DOC_HEIGHT,
    DOC_TTS_PITCH,
    DOC_TTS_RATE,
    DOC_WIDTH,
    DOCUMENTARY_VOICES_EN,
    SHORTS_BEAT_GAP_SECONDS,
    SHORTS_HEIGHT,
    SHORTS_TTS_PITCH,
    SHORTS_TTS_RATE,
    SHORTS_VOICES_EN,
    SHORTS_WIDTH,
    StudioPaths,
)
from .content_engine import (
    build_documentary_description,
    build_short_description,
    generate_documentary_script,
    generate_short_script,
)
from .fact_researcher import research_topic
from .media_engine import fetch_clips_for_queries
from .state_store import StateStore, atomic_write_json, content_hash, normalize_key, read_json
from .subtitle_engine import (
    beats_to_srt_cues,
    generate_documentary_chapters_ass,
    generate_shorts_karaoke_ass,
    write_srt,
)
from .thumbnail_engine import extract_best_frame, generate_thumbnail
from .topic_catalog import pick_next_documentary_topic, pick_next_short_topic
from .video_engine import render_documentary, render_short, verify_rendered_video
from .youtube_engine import (
    add_to_documentary_playlist,
    build_youtube_clients,
    check_remote_duplicate,
    enqueue_video_for_post_processing,
    post_engagement_comment,
    process_subtitle_queue,
    record_reserved_slot,
    reserve_publish_slot,
    set_custom_thumbnail,
    upload_video_resumable,
    verify_channel_ownership,
)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_key(value)).strip("_")[:48] or "video"


def cmd_produce_short(args: argparse.Namespace, paths: StudioPaths, creds: ApiCredentials) -> int:
    paths.ensure_dirs()
    store = StateStore(paths.state_dir)

    topic = pick_next_short_topic(
        store,
        series_override=args.series or "",
        topic_override=args.topic or "",
    )
    print(f"[SHORT] Selected topic: {topic.topic_id} | series={topic.series} | subject={topic.subject}", flush=True)

    sources = research_topic(
        topic.subject,
        wiki_titles=topic.wiki_titles,
        official_urls=topic.official_urls,
        cache_dir=paths.cache_dir,
    )
    print(f"[SHORT] Grounded reference sources: {len(sources)}", flush=True)

    insights = store.get_channel_insights()
    script = generate_short_script(topic, sources, api_key=creds.gemini_api_key, insights=insights)
    digest = content_hash([topic.topic_id, script.title] + [b.narration for b in script.beats])
    print(
        f"[SHORT] Script ready: '{script.title}' ({len(script.beats)} beats, "
        f"{sum(len(b.narration.split()) for b in script.beats)} words)",
        flush=True,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slugify(topic.topic_id)
    run_work = paths.work_dir / f"short_{stamp}_{slug}"
    run_work.mkdir(parents=True, exist_ok=True)

    # 1. Voiceover + exact WordBoundary timings
    narration_wav, beats, audio_duration = synthesize_narration_beats(
        [b.narration for b in script.beats],
        run_work,
        voices=SHORTS_VOICES_EN,
        rate=SHORTS_TTS_RATE,
        pitch=SHORTS_TTS_PITCH,
        gap_seconds=SHORTS_BEAT_GAP_SECONDS,
    )
    print(f"[SHORT] Narration synthesized: {audio_duration:.2f}s across {len(beats)} beats", flush=True)

    # 2. Procedural stereo nature bed + broadcast ducking & -14 LUFS loudness
    ambience_wav = generate_nature_soundscape(
        run_work / "audio" / "ambience.wav",
        audio_duration + 0.5,
        mood=topic.mood,
        include_hook_swell=True,
    )
    mixed_audio = mix_narration_and_ambience(
        narration_wav,
        ambience_wav,
        run_work / "audio" / "mixed.m4a",
        duration_seconds=audio_duration,
        ambience_volume=0.22,
    )

    # 3. Karaoke ASS + English SRT
    ass_path = generate_shorts_karaoke_ass(
        beats,
        run_work / "captions.ass",
        pill_label=script.pill_label,
        width=SHORTS_WIDTH,
        height=SHORTS_HEIGHT,
    )
    srt_cues = beats_to_srt_cues(beats)
    write_srt(srt_cues, paths.output_dir / f"{stamp}_{slug}.en.srt")

    # 4. Portrait-first stock footage fetch
    clips = fetch_clips_for_queries(
        [b.visual_query for b in script.beats],
        orientation="portrait",
        target_clips=8,
        work_dir=run_work,
        creds=creds,
        store=store,
        fallback_queries=topic.visual_queries,
    )
    if len(clips) < 2:
        raise RuntimeError(f"Insufficient stock clips fetched ({len(clips)}). Check PEXELS_API_KEY / PIXABAY_API_KEY.")

    # 5. Pure-FFmpeg 1080x1920 master render + automated QC
    out_mp4 = paths.output_dir / f"{stamp}_{slug}_short.mp4"
    render_short(
        clips,
        mixed_audio,
        ass_path,
        out_mp4,
        run_work,
        total_duration=audio_duration,
        width=SHORTS_WIDTH,
        height=SHORTS_HEIGHT,
    )
    qc = verify_rendered_video(
        out_mp4,
        expected_width=SHORTS_WIDTH,
        expected_height=SHORTS_HEIGHT,
        min_duration=18.0,
        work_dir=run_work,
    )
    print(f"[SHORT] QC PASSED: {qc}", flush=True)

    # 6. High-CTR vertical cover thumbnail
    best_frame = extract_best_frame(out_mp4, run_work)
    thumb_jpg = paths.output_dir / f"{stamp}_{slug}_short.jpg"
    generate_thumbnail(
        best_frame,
        thumb_jpg,
        hook_text=script.thumbnail_hook,
        sub_label=script.pill_label,
        format_type="short",
    )

    description = build_short_description(script, topic, sources)
    manifest: dict[str, Any] = {
        "format": "short",
        "topic_id": topic.topic_id,
        "series": topic.series,
        "title": script.title,
        "thumbnail_hook": script.thumbnail_hook,
        "description": description,
        "pinned_comment": script.pinned_comment,
        "tags": script.tags,
        "qc": qc,
        "video_file": out_mp4.name,
        "thumbnail_file": thumb_jpg.name,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        atomic_write_json(paths.output_dir / f"{stamp}_{slug}_meta.json", manifest)
        print(f"[SHORT] Dry-run complete. Artifacts saved in {paths.output_dir}", flush=True)
        return 0

    # 7. Verified YouTube upload + scheduling + post-processing
    youtube, _ = build_youtube_clients(creds, local_token_path=paths.repo_root / "token.json")
    channel_info = verify_channel_ownership(youtube, creds.expected_channel_id)
    print(f"[YOUTUBE] Channel lock verified: {channel_info['title']} ({channel_info['channel_id']})", flush=True)

    dup_id = check_remote_duplicate(youtube, channel_info["uploads_playlist"], script.title)
    if dup_id:
        print(f"[YOUTUBE] Duplicate already uploaded ({dup_id}); recording state without re-uploading.", flush=True)
        video_id = dup_id
        publish_at = None
        comment_posted = False
    else:
        publish_at = reserve_publish_slot(store, format_type="short", publish_now=args.publish_now)
        video_id = upload_video_resumable(
            youtube,
            out_mp4,
            title=script.title,
            description=description,
            tags=script.tags,
            category_id=topic.category_id,
            publish_at=publish_at,
        )
        print(f"[YOUTUBE] Uploaded Short: https://youtube.com/shorts/{video_id} (publishAt={publish_at or 'PUBLIC'})", flush=True)
        set_custom_thumbnail(youtube, video_id, thumb_jpg)
        comment_posted = False
        if not publish_at:
            comment_posted = post_engagement_comment(youtube, video_id, script.pinned_comment)
        record_reserved_slot(store, publish_at=publish_at, format_type="short", video_id=video_id, title=script.title)

    # Persist durable state
    store.record_topic(
        topic_id=topic.topic_id,
        series=topic.series,
        title=script.title,
        video_id=video_id,
        duration_seconds=qc["duration"],
        format_type="short",
    )
    if topic.location_name:
        store.record_uploaded_location(topic.location_name)
    store.record_used_media_ids([c.asset_key for c in clips])
    store.record_content_hash(digest)
    enqueue_video_for_post_processing(
        store,
        video_id=video_id,
        title=script.title,
        description=description,
        cues=srt_cues,
        pinned_comment=script.pinned_comment,
        comment_posted=comment_posted,
        publish_at=publish_at,
    )

    if not args.skip_i18n:
        i18n_stats = process_subtitle_queue(youtube, store, creds, run_work)
        print(f"[I18N] Subtitle/localization pass: {i18n_stats}", flush=True)

    manifest["video_id"] = video_id
    manifest["publish_at"] = publish_at
    atomic_write_json(paths.output_dir / f"{stamp}_{slug}_meta.json", manifest)
    shutil.rmtree(run_work, ignore_errors=True)
    return 0


def cmd_produce_doc(args: argparse.Namespace, paths: StudioPaths, creds: ApiCredentials) -> int:
    paths.ensure_dirs()
    store = StateStore(paths.state_dir)

    topic = pick_next_documentary_topic(store, place_override=args.place or "")
    print(f"[DOC] Selected documentary: {topic.place} ({topic.region})", flush=True)

    sources = research_topic(
        topic.place,
        wiki_titles=(topic.wiki_title,),
        official_urls=(topic.official_url,) if topic.official_url else (),
        cache_dir=paths.cache_dir,
    )
    print(f"[DOC] Grounded reference sources: {len(sources)}", flush=True)

    insights = store.get_channel_insights()
    script = generate_documentary_script(topic, sources, api_key=creds.gemini_api_key, insights=insights)
    digest = content_hash([topic.slug, script.title] + [c.narration for c in script.chapters])
    print(f"[DOC] Script ready: '{script.title}' ({len(script.chapters)} chapters)", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slugify(topic.slug)
    run_work = paths.work_dir / f"doc_{stamp}_{slug}"
    run_work.mkdir(parents=True, exist_ok=True)

    # 1. Chapter narration synthesis
    narration_wav, beats, audio_duration = synthesize_narration_beats(
        [ch.narration for ch in script.chapters],
        run_work,
        voices=DOCUMENTARY_VOICES_EN,
        rate=DOC_TTS_RATE,
        pitch=DOC_TTS_PITCH,
        gap_seconds=DOC_CHAPTER_GAP_SECONDS,
    )
    print(f"[DOC] Narration synthesized: {audio_duration / 60.0:.2f} min across {len(beats)} chapters", flush=True)

    # 2. Nature ambience bed + ducking + -14 LUFS loudness
    ambience_wav = generate_nature_soundscape(
        run_work / "audio" / "doc_ambience.wav",
        audio_duration + 1.0,
        mood=topic.mood,
        include_hook_swell=False,
    )
    mixed_audio = mix_narration_and_ambience(
        narration_wav,
        ambience_wav,
        run_work / "audio" / "doc_mixed.m4a",
        duration_seconds=audio_duration,
        ambience_volume=0.25,
    )

    # 3. Chapter timestamps, lower-third ASS, and full English SRT
    chapter_timestamps: list[tuple[float, str]] = []
    ass_chapters: list[tuple[float, float, int, str]] = []
    for idx, (ch, beat) in enumerate(zip(script.chapters, beats), start=1):
        chapter_timestamps.append((beat.start, ch.chapter_title))
        ass_chapters.append((beat.start, beat.duration, idx, ch.chapter_title))

    chapters_ass = generate_documentary_chapters_ass(
        ass_chapters,
        run_work / "chapters.ass",
        width=DOC_WIDTH,
        height=DOC_HEIGHT,
    )
    srt_cues = beats_to_srt_cues(beats, max_words_per_cue=14)
    write_srt(srt_cues, paths.output_dir / f"{stamp}_{slug}_doc.en.srt")

    # 4. Landscape footage across Pexels, Pixabay, and Wikimedia Commons
    chapter_queries = [q for ch in script.chapters for q in ch.visual_queries]
    clips = fetch_clips_for_queries(
        chapter_queries,
        orientation="landscape",
        target_clips=20,
        work_dir=run_work,
        creds=creds,
        store=store,
        fallback_queries=topic.visual_queries,
        commons_category=topic.commons_category,
    )
    if len(clips) < 4:
        raise RuntimeError(f"Insufficient landscape clips fetched ({len(clips)}).")

    media_credits = [
        f"{c.provider}: {c.author} ({c.license_name}) {c.page_url}".strip()
        for c in clips
        if c.provider == "Wikimedia Commons" or c.author
    ]

    # 5. Pure-FFmpeg 1920x1080 Documentary render + automated QC
    out_mp4 = paths.output_dir / f"{stamp}_{slug}_doc.mp4"
    render_documentary(
        clips,
        mixed_audio,
        chapters_ass,
        out_mp4,
        run_work,
        total_duration=audio_duration,
        width=DOC_WIDTH,
        height=DOC_HEIGHT,
    )
    qc = verify_rendered_video(
        out_mp4,
        expected_width=DOC_WIDTH,
        expected_height=DOC_HEIGHT,
        min_duration=150.0,
        work_dir=run_work,
    )
    print(f"[DOC] QC PASSED: {qc}", flush=True)

    # 6. High-CTR 1280x720 Documentary Thumbnail
    best_frame = extract_best_frame(out_mp4, run_work, sample_count=9)
    thumb_jpg = paths.output_dir / f"{stamp}_{slug}_doc.jpg"
    generate_thumbnail(
        best_frame,
        thumb_jpg,
        hook_text=script.thumbnail_hook,
        sub_label=topic.place.upper(),
        format_type="documentary",
    )

    description = build_documentary_description(script, topic, chapter_timestamps, sources, media_credits)
    manifest: dict[str, Any] = {
        "format": "documentary",
        "slug": topic.slug,
        "place": topic.place,
        "title": script.title,
        "thumbnail_hook": script.thumbnail_hook,
        "chapters": [{"start": round(s, 1), "title": t} for s, t in chapter_timestamps],
        "qc": qc,
        "video_file": out_mp4.name,
        "thumbnail_file": thumb_jpg.name,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        atomic_write_json(paths.output_dir / f"{stamp}_{slug}_doc_meta.json", manifest)
        print(f"[DOC] Dry-run complete. Artifacts saved in {paths.output_dir}", flush=True)
        return 0

    youtube, _ = build_youtube_clients(creds, local_token_path=paths.repo_root / "token.json")
    channel_info = verify_channel_ownership(youtube, creds.expected_channel_id)
    print(f"[YOUTUBE] Channel lock verified: {channel_info['title']} ({channel_info['channel_id']})", flush=True)

    dup_id = check_remote_duplicate(youtube, channel_info["uploads_playlist"], script.title)
    if dup_id:
        video_id = dup_id
        publish_at = None
        comment_posted = False
    else:
        publish_at = reserve_publish_slot(store, format_type="documentary", publish_now=args.publish_now)
        video_id = upload_video_resumable(
            youtube,
            out_mp4,
            title=script.title,
            description=description,
            tags=script.tags,
            category_id=CATEGORY_TRAVEL_EVENTS,
            publish_at=publish_at,
        )
        print(f"[YOUTUBE] Uploaded Documentary: https://youtube.com/watch?v={video_id} (publishAt={publish_at or 'PUBLIC'})", flush=True)
        set_custom_thumbnail(youtube, video_id, thumb_jpg)
        add_to_documentary_playlist(youtube, video_id)
        comment_posted = False
        if not publish_at:
            comment_posted = post_engagement_comment(youtube, video_id, script.pinned_comment)
        record_reserved_slot(store, publish_at=publish_at, format_type="documentary", video_id=video_id, title=script.title)

    store.record_documentary(topic.place, video_id, script.title, qc["duration"])
    store.record_topic(
        topic_id=f"doc_{topic.slug}",
        series="DOCUMENTARY",
        title=script.title,
        video_id=video_id,
        duration_seconds=qc["duration"],
        format_type="documentary",
    )
    store.record_used_media_ids([c.asset_key for c in clips])
    store.record_content_hash(digest)
    enqueue_video_for_post_processing(
        store,
        video_id=video_id,
        title=script.title,
        description=description,
        cues=srt_cues,
        pinned_comment=script.pinned_comment,
        comment_posted=comment_posted,
        publish_at=publish_at,
    )

    if not args.skip_i18n:
        i18n_stats = process_subtitle_queue(youtube, store, creds, run_work)
        print(f"[I18N] Subtitle/localization pass: {i18n_stats}", flush=True)

    manifest["video_id"] = video_id
    manifest["publish_at"] = publish_at
    atomic_write_json(paths.output_dir / f"{stamp}_{slug}_doc_meta.json", manifest)
    shutil.rmtree(run_work, ignore_errors=True)
    return 0


def cmd_optimize_analytics(args: argparse.Namespace, paths: StudioPaths, creds: ApiCredentials) -> int:
    paths.ensure_dirs()
    store = StateStore(paths.state_dir)
    youtube, analytics = build_youtube_clients(creds, local_token_path=paths.repo_root / "token.json")
    channel_info = verify_channel_ownership(youtube, creds.expected_channel_id)
    insights = run_analytics_optimization(
        youtube,
        analytics,
        store,
        channel_info,
        lookback_days=args.lookback_days,
    )
    print(json.dumps(insights["rolling_30d"], indent=2), flush=True)
    print(f"[ANALYTICS] Updated series weights: {insights['series_weights']}", flush=True)
    return 0


def cmd_backfill_subtitles(args: argparse.Namespace, paths: StudioPaths, creds: ApiCredentials) -> int:
    paths.ensure_dirs()
    store = StateStore(paths.state_dir)
    youtube, _ = build_youtube_clients(creds, local_token_path=paths.repo_root / "token.json")
    verify_channel_ownership(youtube, creds.expected_channel_id)
    stats = process_subtitle_queue(youtube, store, creds, paths.work_dir)
    print(f"[I18N] Backfill complete: {stats}", flush=True)
    return 0


def cmd_verify_auth(args: argparse.Namespace, paths: StudioPaths, creds: ApiCredentials) -> int:
    youtube, _ = build_youtube_clients(creds, local_token_path=paths.repo_root / "token.json")
    info = verify_channel_ownership(youtube, creds.expected_channel_id)
    print(
        f"[AUTH OK] Channel '{info['title']}' ({info['channel_id']}) | "
        f"subs={info['subscribers']} | views={info['views']} | videos={info['video_count']}",
        flush=True,
    )
    return 0


def migrate_v1_state(v1_dir: Path, state_dir: Path) -> dict[str, int]:
    """Import clean deduplication & archive history from legacy v1 CampShorts without paths or secrets."""
    store = StateStore(state_dir)
    imported_locations = 0
    seen_loc: set[str] = {normalize_key(x) for x in store.get_uploaded_locations()}
    merged_locs = store.get_uploaded_locations()

    for fname in ("yuklenen_videolar.txt", "gecmis_hafiza.txt"):
        p = v1_dir / fname
        if not p.is_file():
            continue
        for raw_line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            loc = raw_line.split("|", 1)[0].strip()
            key = normalize_key(loc)
            if loc and key and key not in seen_loc:
                seen_loc.add(key)
                merged_locs.append(loc)
                imported_locations += 1
    atomic_write_json(
        state_dir / "uploaded_locations.json",
        {"updated_at": datetime.now(timezone.utc).isoformat(), "locations": merged_locs},
    )

    media_raw = read_json(v1_dir / "media_hafiza.json", [])
    media_ids = [str(m).strip() for m in media_raw if isinstance(m, str) and m.strip()] if isinstance(media_raw, list) else []
    store.record_used_media_ids(media_ids)

    hashes_raw = read_json(v1_dir / "icerik_hafiza.json", [])
    clean_hashes = [str(h).strip() for h in hashes_raw if isinstance(h, str) and len(h.strip()) == 64] if isinstance(hashes_raw, list) else []
    merged_hashes = sorted(store.get_content_hashes() | set(clean_hashes))
    atomic_write_json(
        state_dir / "content_hashes.json",
        {"updated_at": datetime.now(timezone.utc).isoformat(), "hashes": merged_hashes},
    )

    # Import legacy documentaries while strictly stripping any local Windows file paths!
    legacy_docs = read_json(v1_dir / "belgesel_arsivi.json", {})
    archive = store.get_documentary_archive()
    docs_imported = 0
    if isinstance(legacy_docs, dict):
        for slug, row in legacy_docs.items():
            if not isinstance(row, dict):
                continue
            clean_slug = normalize_key(slug).replace(" ", "_")
            if clean_slug and clean_slug not in archive:
                archive[clean_slug] = {
                    "place": str(row.get("place") or slug),
                    "video_id": str(row.get("video_id") or ""),
                    "legacy_date": str(row.get("date") or ""),
                    "status": str(row.get("status") or "LEGACY"),
                }
                docs_imported += 1
    atomic_write_json(state_dir / "documentary_archive.json", archive)

    # Seed baseline analytics insights cleanly if channel_insights.json doesn't exist yet
    if not (state_dir / "channel_insights.json").is_file():
        growth = read_json(v1_dir / "KANAL_BUYUME_DURUM.json", {})
        totals = growth.get("channel_totals", {}) if isinstance(growth, dict) else {}
        baseline = {
            "source": "migrated_v1_baseline",
            "series_weights": dict(DEFAULT_SERIES_WEIGHTS),
            "rolling_30d": {
                "views": int(totals.get("views", 0) or 0),
                "avg_view_percentage": round(float(totals.get("averageViewPercentage", 47.5) or 47.5), 1),
                "net_subscribers": int(growth.get("net_subscribers", 0) or 0),
            },
            "summary_guidance": (
                "Baseline v1 Shorts averaged 47.5% AVP due to short encyclopedia-style narration and slow visual cuts. "
                "Prioritize instant 2-second curiosity hooks, 35-48s escalating narratives, and bold survival/wild-place takeaways."
            ),
        }
        store.save_channel_insights(baseline)

    return {
        "locations": len(merged_locs),
        "media_ids": len(store.get_used_media_ids()),
        "content_hashes": len(merged_hashes),
        "documentaries": len(archive),
        "newly_imported_locations": imported_locations,
        "newly_imported_docs": docs_imported,
    }


def cmd_migrate_state(args: argparse.Namespace, paths: StudioPaths, creds: ApiCredentials) -> int:
    paths.ensure_dirs()
    v1_dir = Path(args.from_v1).resolve()
    if not v1_dir.is_dir():
        raise FileNotFoundError(f"v1 directory not found: {v1_dir}")
    summary = migrate_v1_state(v1_dir, paths.state_dir)
    print(f"[MIGRATE] Clean v1 state imported: {json.dumps(summary)}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="campuncharted", description="CampUncharted Studio v2 autonomous pipeline")
    parser.add_argument("--repo-root", default=None, help="Repository root directory (defaults to CWD)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_short = sub.add_parser("produce-short", help="Generate and publish one viral 9:16 Short")
    p_short.add_argument("--dry-run", action="store_true", help="Render & QC video without uploading to YouTube")
    p_short.add_argument("--series", default="", help="Override series (e.g. CAMP_MISTAKES, WILD_STAYS)")
    p_short.add_argument("--topic", default="", help="Override topic ID or custom subject")
    p_short.add_argument("--publish-now", action="store_true", help="Publish immediately instead of peak UTC slot")
    p_short.add_argument("--skip-i18n", action="store_true", help="Skip immediate multi-language subtitle pass")
    p_short.set_defaults(func=cmd_produce_short)

    p_doc = sub.add_parser("produce-doc", help="Generate and publish one 6-10 min 16:9 Documentary")
    p_doc.add_argument("--dry-run", action="store_true", help="Render & QC documentary without uploading")
    p_doc.add_argument("--place", default="", help="Override documentary destination slug or name")
    p_doc.add_argument("--publish-now", action="store_true", help="Publish immediately instead of 16:00 UTC slot")
    p_doc.add_argument("--skip-i18n", action="store_true", help="Skip immediate subtitle pass")
    p_doc.set_defaults(func=cmd_produce_doc)

    p_ana = sub.add_parser("optimize-analytics", help="Fetch YouTube Analytics & rebalance series weights")
    p_ana.add_argument("--lookback-days", type=int, default=30)
    p_ana.set_defaults(func=cmd_optimize_analytics)

    p_sub = sub.add_parser("backfill-subtitles", help="Process quota-safe multi-language subtitles & comments")
    p_sub.set_defaults(func=cmd_backfill_subtitles)

    p_auth = sub.add_parser("verify-auth", help="Verify YouTube OAuth token & channel lock")
    p_auth.set_defaults(func=cmd_verify_auth)

    p_mig = sub.add_parser("migrate-state", help="Import clean deduplication state from legacy CampShorts v1")
    p_mig.add_argument("--from-v1", required=True, help="Path to unpacked legacy CampShorts folder")
    p_mig.set_defaults(func=cmd_migrate_state)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = StudioPaths.from_root(args.repo_root)
    creds = ApiCredentials()
    return int(args.func(args, paths, creds))


if __name__ == "__main__":
    sys.exit(main())
