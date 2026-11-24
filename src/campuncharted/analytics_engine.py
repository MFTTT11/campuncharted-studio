"""YouTube Analytics feedback loop that measures retention/subs and rebalances series weights."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from googleapiclient.errors import HttpError

from .config import DEFAULT_SERIES_WEIGHTS
from .state_store import StateStore


def rebalance_series_weights(
    series_metrics: dict[str, dict[str, float]],
    current_weights: dict[str, float] | None = None,
    *,
    min_weight: float = 0.08,
    max_weight: float = 0.45,
) -> dict[str, float]:
    """Softly rebalance series weights based on measured AVP, views, and subscriber gain.

    Pure function so it can be unit-tested thoroughly without API calls.
    """
    base = dict(DEFAULT_SERIES_WEIGHTS)
    if current_weights:
        for k, v in current_weights.items():
            if k in base and isinstance(v, (int, float)) and v > 0:
                base[k] = float(v)

    measured_scores: dict[str, float] = {}
    for series_name, stats in series_metrics.items():
        if series_name not in base:
            continue
        count = int(stats.get("count", 0) or 0)
        if count < 2:
            continue
        avg_views = max(1.0, float(stats.get("avg_views", 0.0)))
        avg_avp = max(10.0, min(150.0, float(stats.get("avg_avp", 50.0))))
        net_subs = float(stats.get("net_subs", 0.0))
        # Reward retention (AVP) strongly, log-scaled views, and positive sub conversion
        score = (avg_avp / 60.0) * math.log10(avg_views + 10.0) * (1.0 + max(-0.25, min(1.0, net_subs * 0.15)))
        measured_scores[series_name] = max(0.1, score)

    if len(measured_scores) < 2:
        total = sum(base.values()) or 1.0
        return {k: round(v / total, 4) for k, v in base.items()}

    # Blend measured distribution gently (30% learning rate) so the channel adapts without oscillating
    score_sum = sum(measured_scores.values())
    measured_share_budget = sum(base[s] for s in measured_scores)
    updated = dict(base)
    for s_name, sc in measured_scores.items():
        target = (sc / score_sum) * measured_share_budget
        updated[s_name] = (0.70 * base[s_name]) + (0.30 * target)

    # Clamp & renormalize iteratively
    for _ in range(4):
        for k in updated:
            updated[k] = max(min_weight, min(max_weight, updated[k]))
        tot = sum(updated.values()) or 1.0
        updated = {k: v / tot for k, v in updated.items()}

    return {k: round(v, 4) for k, v in updated.items()}


def _rows_to_dicts(report: dict[str, Any]) -> list[dict[str, Any]]:
    headers = [h.get("name", "") for h in report.get("columnHeaders", [])]
    rows = report.get("rows") or []
    return [dict(zip(headers, row)) for row in rows]


def _fetch_video_titles(youtube: Any, video_ids: list[str]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        if not batch:
            continue
        try:
            resp = youtube.videos().list(part="snippet", id=",".join(batch)).execute()
            for item in resp.get("items", []):
                titles[str(item.get("id", ""))] = str(item.get("snippet", {}).get("title", ""))
        except HttpError:
            continue
    return titles


def run_analytics_optimization(
    youtube: Any,
    analytics: Any,
    store: StateStore,
    channel_info: dict[str, Any],
    *,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Query YouTube Analytics (last 30 days), evaluate series performance, and update insights."""
    end_date = (datetime.now(timezone.utc) - timedelta(days=2)).date()
    start_date = end_date - timedelta(days=max(7, lookback_days))
    start_s = start_date.isoformat()
    end_s = end_date.isoformat()

    totals_report = (
        analytics.reports()
        .query(
            ids="channel==MINE",
            startDate=start_s,
            endDate=end_s,
            metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost,likes,comments",
        )
        .execute()
    )
    totals_rows = _rows_to_dicts(totals_report)
    totals = totals_rows[0] if totals_rows else {}

    per_video_report = (
        analytics.reports()
        .query(
            ids="channel==MINE",
            startDate=start_s,
            endDate=end_s,
            dimensions="video",
            metrics="views,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost,likes,comments",
            sort="-views",
            maxResults=50,
        )
        .execute()
    )
    video_rows = _rows_to_dicts(per_video_report)
    video_ids = [str(r.get("video", "")) for r in video_rows if r.get("video")]
    titles = _fetch_video_titles(youtube, video_ids)

    # Map known v2 videos to their series
    vid_to_series = {
        str(row.get("video_id", "")): str(row.get("series", ""))
        for row in store.get_topic_history()
        if row.get("video_id") and row.get("series")
    }

    series_agg: dict[str, dict[str, float]] = {}
    top_videos: list[dict[str, Any]] = []
    for row in video_rows:
        vid = str(row.get("video", ""))
        views = float(row.get("views", 0) or 0)
        avp = float(row.get("averageViewPercentage", 0) or 0)
        avd = float(row.get("averageViewDuration", 0) or 0)
        net_subs = float(row.get("subscribersGained", 0) or 0) - float(row.get("subscribersLost", 0) or 0)
        series = vid_to_series.get(vid, "")

        if len(top_videos) < 8:
            top_videos.append(
                {
                    "video_id": vid,
                    "title": titles.get(vid, vid),
                    "series": series or "LEGACY",
                    "views": int(views),
                    "avp": round(avp, 1),
                    "avd": round(avd, 1),
                    "net_subs": int(net_subs),
                }
            )

        if series:
            bucket = series_agg.setdefault(series, {"count": 0.0, "views": 0.0, "avp_sum": 0.0, "net_subs": 0.0})
            bucket["count"] += 1.0
            bucket["views"] += views
            bucket["avp_sum"] += avp
            bucket["net_subs"] += net_subs

    series_metrics: dict[str, dict[str, float]] = {}
    for s_name, agg in series_agg.items():
        cnt = max(1.0, agg["count"])
        series_metrics[s_name] = {
            "count": int(agg["count"]),
            "avg_views": round(agg["views"] / cnt, 1),
            "avg_avp": round(agg["avp_sum"] / cnt, 1),
            "net_subs": int(agg["net_subs"]),
        }

    existing_insights = store.get_channel_insights()
    new_weights = rebalance_series_weights(series_metrics, existing_insights.get("series_weights"))

    avg_avp_all = float(totals.get("averageViewPercentage", 0) or 0)
    total_views_30d = int(totals.get("views", 0) or 0)
    net_subs_30d = int(totals.get("subscribersGained", 0) or 0) - int(totals.get("subscribersLost", 0) or 0)

    best_series_note = ""
    if series_metrics:
        best_s = max(series_metrics.items(), key=lambda kv: (kv[1]["avg_avp"], kv[1]["avg_views"]))
        best_series_note = f" Top-performing series right now: {best_s[0]} ({best_s[1]['avg_avp']:.0f}% avg AVP, {best_s[1]['avg_views']:.0f} avg views)."

    guidance = (
        f"Last {lookback_days}d channel performance: {total_views_30d} views, {avg_avp_all:.1f}% average view percentage, "
        f"{net_subs_30d:+d} net subscribers.{best_series_note} "
        "Prioritize immediate 2-second visual/curiosity hooks, high-contrast payoff beats, and save-worthy survival or place revelations."
    )

    existing_trends = [
        str(p).strip()
        for p in (existing_insights.get("live_youtube_trends") or [])
        if str(p).strip()
    ][:4]
    if existing_trends:
        guidance = f"{guidance} Live YouTube trends: {' | '.join(existing_trends[:2])}".strip()

    insights: dict[str, Any] = {
        "period": {"start": start_s, "end": end_s},
        "channel_stats": {
            "subscribers": channel_info.get("subscribers", 0),
            "total_views": channel_info.get("views", 0),
            "video_count": channel_info.get("video_count", 0),
        },
        "rolling_30d": {
            "views": total_views_30d,
            "watch_minutes": round(float(totals.get("estimatedMinutesWatched", 0) or 0), 1),
            "avg_view_duration_sec": round(float(totals.get("averageViewDuration", 0) or 0), 1),
            "avg_view_percentage": round(avg_avp_all, 1),
            "subscribers_gained": int(totals.get("subscribersGained", 0) or 0),
            "subscribers_lost": int(totals.get("subscribersLost", 0) or 0),
            "net_subscribers": net_subs_30d,
            "likes": int(totals.get("likes", 0) or 0),
            "comments": int(totals.get("comments", 0) or 0),
        },
        "series_metrics": series_metrics,
        "series_weights": new_weights,
        "top_videos": top_videos,
        "summary_guidance": guidance,
    }
    if existing_trends:
        insights["live_youtube_trends"] = existing_trends
    store.save_channel_insights(insights)
    store.append_analytics_snapshot(
        {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "subscribers": channel_info.get("subscribers", 0),
            "total_views": channel_info.get("views", 0),
            "views_30d": total_views_30d,
            "avp_30d": round(avg_avp_all, 1),
            "net_subs_30d": net_subs_30d,
        }
    )
    return insights
