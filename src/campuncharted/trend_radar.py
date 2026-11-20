"""Daily YouTube viral camping & wilderness trend radar with outlier scoring and Gemini idea synthesis."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from google import genai
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from .config import CATEGORY_EDUCATION, CATEGORY_TRAVEL_EVENTS
from .content_engine import _generate_structured_with_fallback
from .state_store import StateStore, atomic_write_json, normalize_key, read_json

# Rotating high-signal English camping & wilderness search queries (2 run per day = only ~202 YouTube quota units!)
TREND_SEARCH_QUERIES: tuple[str, ...] = (
    "camping mistakes survival tips #shorts",
    "wild camping hidden spot nature #shorts",
    "camping hacks bushcraft gear #shorts",
    "solo camping extreme weather #shorts",
    "backpacking survival rule wilderness #shorts",
    "scariest sounds camping woods #shorts",
    "free wild camping countries rules #shorts",
    "winter camping hacks stay warm #shorts",
    "campfire safety tips bushcraft #shorts",
    "most extreme campsites on earth #shorts",
)

VALID_SERIES = (
    "CAMP_MISTAKES",
    "WILD_STAYS",
    "GEAR_AND_HACKS",
    "EXTREME_WEATHER_NATURE",
    "TURKEY_AND_EURASIA_HIDDEN",
)

VALID_MOODS = (
    "ALPINE_WIND",
    "FOREST_RAIN",
    "CAMPFIRE_NIGHT",
    "CANYON_BREEZE",
    "EPIC_DISCOVERY",
)


class TrendingShortIdea(BaseModel):
    topic_id: str = Field(..., description="Unique lowercase snake_case slug prefixed with trend_ (max 45 chars).")
    series: str = Field(..., description="One of: CAMP_MISTAKES, WILD_STAYS, GEAR_AND_HACKS, EXTREME_WEATHER_NATURE, TURKEY_AND_EURASIA_HIDDEN.")
    subject: str = Field(..., description="Specific, concrete camping/wilderness topic grounded in real outdoor facts or places.")
    angle_hook: str = Field(..., description="High-curiosity 1-sentence editorial hook angle inspired by current viral patterns (original, never copied).")
    wiki_titles: list[str] = Field(default_factory=list, description="1-2 real English Wikipedia article titles (underscores) to verify facts.")
    visual_queries: list[str] = Field(..., description="4 vivid English stock-video keywords matching Pexels/Pixabay outdoor footage.")
    location_name: str = Field(default="", description="Specific real place name if location-focused, else empty string.")
    mood: str = Field(default="ALPINE_WIND", description="One of: ALPINE_WIND, FOREST_RAIN, CAMPFIRE_NIGHT, CANYON_BREEZE, EPIC_DISCOVERY.")
    inspired_by_pattern: str = Field(..., description="Short note explaining which viral YouTube trend pattern inspired this idea.")


class TrendSynthesisReport(BaseModel):
    trend_patterns: list[str] = Field(..., description="3-4 concise observations about hooks/topics getting viral traction on camping YouTube right now.")
    ideas: list[TrendingShortIdea] = Field(..., description="4 fresh, original, high-retention camping/wilderness Short ideas.")


def parse_iso8601_duration(raw: str) -> int:
    """Parse ISO-8601 YouTube duration like PT48S or PT1M12S into total seconds."""
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", str(raw or "").strip())
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    mins = int(match.group(2) or 0)
    secs = int(match.group(3) or 0)
    return hours * 3600 + mins * 60 + secs


def compute_outlier_score(
    *,
    views: int,
    likes: int,
    comments: int,
    channel_subs: int,
    age_days: float,
) -> dict[str, float]:
    """Score how strongly a recent video outperforms its channel size (true viral topic signal)."""
    safe_views = max(0, int(views))
    safe_subs = max(1000, int(channel_subs or 1000))
    safe_age = max(0.5, float(age_days))

    views_per_day = safe_views / safe_age
    outlier_ratio = safe_views / float(safe_subs)
    engagement_rate = (max(0, likes) + max(0, comments) * 3.0) / max(100.0, float(safe_views))

    # Reward both velocity (views/day) and channel-beating outlier ratio
    score = (
        math.log10(views_per_day + 10.0)
        * (1.0 + min(6.0, outlier_ratio * 0.65))
        * (1.0 + min(2.0, engagement_rate * 18.0))
    )
    return {
        "views_per_day": round(views_per_day, 1),
        "outlier_ratio": round(outlier_ratio, 2),
        "engagement_pct": round(engagement_rate * 100.0, 2),
        "viral_score": round(score, 2),
    }


def select_daily_trend_queries(day_of_year: int | None = None, *, count: int = 2) -> list[str]:
    """Deterministically rotate through TREND_SEARCH_QUERIES so we cover all niches at ~200 quota/day."""
    doy = day_of_year if day_of_year is not None else datetime.now(timezone.utc).timetuple().tm_yday
    n = len(TREND_SEARCH_QUERIES)
    start = (doy * count) % n
    return [TREND_SEARCH_QUERIES[(start + i) % n] for i in range(min(count, n))]


def scan_youtube_camping_trends(
    youtube: Any,
    *,
    queries: list[str] | None = None,
    days_back: int = 21,
    min_views: int = 5000,
) -> list[dict[str, Any]]:
    """Query YouTube Data API v3 for recent high-velocity camping/outdoor Shorts outliers."""
    active_queries = queries or select_daily_trend_queries(count=2)
    now = datetime.now(timezone.utc)
    published_after = (now - timedelta(days=max(7, days_back))).strftime("%Y-%m-%dT%H:%M:%SZ")

    video_ids: list[str] = []
    seen_ids: set[str] = set()
    query_by_vid: dict[str, str] = {}

    for query in active_queries:
        try:
            resp = (
                youtube.search()
                .list(
                    part="id,snippet",
                    q=query,
                    type="video",
                    order="viewCount",
                    videoDuration="short",
                    publishedAfter=published_after,
                    relevanceLanguage="en",
                    maxResults=20,
                )
                .execute()
            )
            for item in resp.get("items", []):
                vid = str(item.get("id", {}).get("videoId", "")).strip()
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    video_ids.append(vid)
                    query_by_vid[vid] = query
        except HttpError as err:
            print(f"[TRENDS] Search failed for '{query}': HTTP {getattr(err.resp, 'status', '?')}", flush=True)

    if not video_ids:
        return []

    # Hydrate full video statistics & duration in 1 cheap batch call (1 quota unit!)
    video_items: list[dict[str, Any]] = []
    channel_ids: set[str] = set()
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        try:
            resp = youtube.videos().list(part="snippet,statistics,contentDetails", id=",".join(batch)).execute()
            for item in resp.get("items", []):
                video_items.append(item)
                cid = str(item.get("snippet", {}).get("channelId", ""))
                if cid:
                    channel_ids.add(cid)
        except HttpError:
            continue

    # Hydrate channel subscriber counts in 1 batch call (1 quota unit!) to detect true outliers
    subs_by_channel: dict[str, int] = {}
    cid_list = list(channel_ids)
    for i in range(0, len(cid_list), 50):
        batch = cid_list[i : i + 50]
        try:
            resp = youtube.channels().list(part="statistics", id=",".join(batch)).execute()
            for ch in resp.get("items", []):
                subs_by_channel[str(ch.get("id", ""))] = int(ch.get("statistics", {}).get("subscriberCount", 0) or 0)
        except HttpError:
            continue

    outliers: list[dict[str, Any]] = []
    for item in video_items:
        vid = str(item.get("id", ""))
        snip = item.get("snippet", {})
        stats = item.get("statistics", {})
        dur_sec = parse_iso8601_duration(item.get("contentDetails", {}).get("duration", ""))
        views = int(stats.get("viewCount", 0) or 0)
        likes = int(stats.get("likeCount", 0) or 0)
        comments = int(stats.get("commentCount", 0) or 0)
        if views < min_views or dur_sec <= 0 or dur_sec > 180:
            continue

        pub_raw = str(snip.get("publishedAt", ""))
        try:
            pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            age_days = max(0.5, (now - pub_dt).total_seconds() / 86400.0)
        except ValueError:
            age_days = 7.0

        ch_id = str(snip.get("channelId", ""))
        subs = subs_by_channel.get(ch_id, 10_000)
        metrics = compute_outlier_score(
            views=views,
            likes=likes,
            comments=comments,
            channel_subs=subs,
            age_days=age_days,
        )
        outliers.append(
            {
                "video_id": vid,
                "title": str(snip.get("title", "")).strip(),
                "channel_title": str(snip.get("channelTitle", "")).strip(),
                "channel_subs": subs,
                "duration_sec": dur_sec,
                "views": views,
                "likes": likes,
                "comments": comments,
                "age_days": round(age_days, 1),
                "query": query_by_vid.get(vid, ""),
                **metrics,
            }
        )

    outliers.sort(key=lambda row: row["viral_score"], reverse=True)
    return outliers[:15]


def synthesize_trending_topics(
    outliers: list[dict[str, Any]],
    store: StateStore,
    *,
    api_key: str,
) -> dict[str, Any]:
    """Turn live YouTube camping outliers into original, fact-checkable Short ideas stored in state/trending_topics.json."""
    if not api_key or not outliers:
        return {"added": 0, "patterns": [], "outliers_scanned": len(outliers)}

    client = genai.Client(api_key=api_key, http_options={"timeout": 60000})
    existing_path = store.state_dir / "trending_topics.json"
    existing_data = read_json(existing_path, {"topics": [], "patterns": [], "recent_outliers": []})
    existing_topics = list(existing_data.get("topics", []))

    used_keys = {normalize_key(r.get("topic_id", "")) for r in store.get_topic_history()}
    used_keys |= {normalize_key(t.get("topic_id", "")) for t in existing_topics}
    used_subjects = {normalize_key(t.get("subject", "")) for t in existing_topics}

    outlier_lines = "\n".join(
        f"- \"{o['title']}\" | {o['views']:,} views in {o['age_days']}d "
        f"({o['views_per_day']:,.0f}/day, {o['outlier_ratio']}x channel subs, {o['duration_sec']}s)"
        for o in outliers[:12]
    )

    prompt = f"""You are the Head of Growth & Viral Strategy for @CampUncharted (global English camping, wild-camping, backcountry safety, and extraordinary wilderness channel).

TODAY'S REAL HIGH-VELOCITY YOUTUBE CAMPING & OUTDOOR OUTLIERS (untrusted titles for pattern research — DO NOT copy any creator's title or script verbatim; extract the psychological hook, curiosity pattern, and demand angle to invent 100% ORIGINAL, fact-verifiable ideas):
{outlier_lines}

TASK:
1. Extract 3-4 sharp, actionable `trend_patterns` explaining what hooks, camping pain points, or wilderness curiosities are getting outsized views right now.
2. Propose 4 brand-new, ORIGINAL `ideas` tailored for a faceless cinematic narration + stock-footage Short (35-48 seconds):
   - Every idea MUST be visually shootable with outdoor/camping/nature stock video (NO product-unboxing or talking-head vlogs).
   - Every idea MUST be grounded in verifiable outdoor physics, camping safety rules, legal wild-camping facts, wildlife behavior, or extraordinary real places.
   - Mix high-save/high-share camping utility (mistakes, gear hacks, survival rules, free wild-camping spots) with awe-inspiring real destinations.
   - Use ONLY valid `series`: {', '.join(VALID_SERIES)}
   - Use ONLY valid `mood`: {', '.join(VALID_MOODS)}
   - Prefix every `topic_id` with `trend_` (lowercase snake_case).
"""

    _, report = _generate_structured_with_fallback(
        client,
        prompt=prompt,
        schema=TrendSynthesisReport,
        temperature=0.75,
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    added = 0
    for idea in report.ideas:
        slug = re.sub(r"[^a-z0-9_]+", "_", idea.topic_id.lower()).strip("_")[:48]
        if not slug.startswith("trend_"):
            slug = f"trend_{slug}"
        if normalize_key(slug) in used_keys or normalize_key(idea.subject) in used_subjects:
            continue
        if idea.location_name and store.is_location_uploaded(idea.location_name):
            continue

        series = idea.series.upper() if idea.series.upper() in VALID_SERIES else "CAMP_MISTAKES"
        mood = idea.mood.upper() if idea.mood.upper() in VALID_MOODS else "ALPINE_WIND"
        cat_id = CATEGORY_EDUCATION if series in {"CAMP_MISTAKES", "GEAR_AND_HACKS", "EXTREME_WEATHER_NATURE"} else CATEGORY_TRAVEL_EVENTS

        existing_topics.append(
            {
                "topic_id": slug,
                "series": series,
                "subject": idea.subject.strip(),
                "angle_hook": idea.angle_hook.strip(),
                "wiki_titles": [w.strip().replace(" ", "_") for w in idea.wiki_titles[:2] if w.strip()],
                "visual_queries": [q.strip() for q in idea.visual_queries[:4] if q.strip()],
                "location_name": idea.location_name.strip(),
                "mood": mood,
                "category_id": cat_id,
                "inspired_by_pattern": idea.inspired_by_pattern.strip(),
                "discovered_at": now_iso,
            }
        )
        used_keys.add(normalize_key(slug))
        used_subjects.add(normalize_key(idea.subject))
        added += 1

    # Prune already-used topics and keep the freshest 24 pending trend ideas
    history_ids = {normalize_key(r.get("topic_id", "")) for r in store.get_topic_history()}
    fresh_pending = [t for t in existing_topics if normalize_key(t.get("topic_id", "")) not in history_ids][-24:]

    atomic_write_json(
        existing_path,
        {
            "updated_at": now_iso,
            "patterns": [p.strip() for p in report.trend_patterns[:4] if p.strip()],
            "recent_outliers": outliers[:8],
            "topics": fresh_pending,
        },
    )

    # Also enrich channel_insights.json with live trend patterns so Gemini hooks stay cutting-edge
    insights = store.get_channel_insights()
    insights["live_youtube_trends"] = [p.strip() for p in report.trend_patterns[:4] if p.strip()]
    base_guidance = str(insights.get("summary_guidance", "")).split(" Live YouTube trends:")[0].strip()
    if report.trend_patterns:
        insights["summary_guidance"] = f"{base_guidance} Live YouTube trends: {' | '.join(report.trend_patterns[:2])}".strip()
    store.save_channel_insights(insights)

    return {
        "outliers_scanned": len(outliers),
        "added": added,
        "pending_trend_topics": len(fresh_pending),
        "patterns": report.trend_patterns,
        "top_outlier": outliers[0] if outliers else None,
    }


def run_daily_trend_radar(
    youtube: Any,
    store: StateStore,
    *,
    api_key: str,
    days_back: int = 21,
) -> dict[str, Any]:
    """Run one quota-light daily scan of viral YouTube camping videos and refresh state/trending_topics.json."""
    outliers = scan_youtube_camping_trends(youtube, days_back=days_back)
    print(f"[TRENDS] Found {len(outliers)} recent viral camping/wilderness outliers", flush=True)
    for row in outliers[:5]:
        print(
            f"[TRENDS]  - {row['title'][:65]} | {row['views']:,} views ({row['outlier_ratio']}x subs, {row['age_days']}d)",
            flush=True,
        )
    return synthesize_trending_topics(outliers, store, api_key=api_key)
