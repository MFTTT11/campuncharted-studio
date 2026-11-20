"""YouTube Data v3 & Analytics v2 engine with HTTP-308 resumable fix, channel lock, and quota-safe i18n."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import google_auth_httplib2
import httplib2
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from .config import (
    ApiCredentials,
    DAILY_LANGUAGE_QUOTA_BUDGET,
    DOC_UTC_SLOTS,
    EXPECTED_CHANNEL_ID,
    SHORTS_UTC_SLOTS,
    SUBTITLE_LANGUAGES,
)
from .content_engine import translate_subtitles_and_metadata
from .state_store import StateStore, normalize_key
from .subtitle_engine import write_srt

RETRIABLE_STATUS_CODES = {500, 502, 503, 504}
CAPTION_INSERT_QUOTA_COST = 400
LOCALIZATION_UPDATE_QUOTA_COST = 50


def _decode_json_secret(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        return json.loads(text)
    # Handle full OAuth Playground HTTP response copy-paste ("HTTP/1.1 200 OK ... { ... }")
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except ValueError:
            pass
    # Handle bare Google OAuth refresh_token string pasted directly ("1//...")
    if text.startswith("1//") and "\n" not in text and " " not in text:
        return {"token": "", "refresh_token": text}
    # Support base64-encoded JSON secrets for safe CI/CD pasting
    decoded = base64.b64decode(text).decode("utf-8")
    return json.loads(decoded)


def load_youtube_credentials(creds: ApiCredentials, *, local_token_path: Path | None = None) -> Credentials:
    """Load and auto-refresh YouTube OAuth credentials without interactive browser login."""
    token_data: dict[str, Any] = {}
    if creds.youtube_token_json:
        token_data = _decode_json_secret(creds.youtube_token_json)
    elif local_token_path and local_token_path.is_file():
        token_data = json.loads(local_token_path.read_text(encoding="utf-8"))
    else:
        raise RuntimeError(
            "YOUTUBE_TOKEN_JSON secret (or local token.json) is missing. "
            "Add your YouTube OAuth token JSON to GitHub Repository Secrets."
        )

    # Merge client_id/client_secret from optional YOUTUBE_CLIENT_SECRET_JSON if omitted in token
    if (not token_data.get("client_id") or not token_data.get("client_secret")) and creds.youtube_client_secret_json:
        secret_blob = _decode_json_secret(creds.youtube_client_secret_json)
        inner = secret_blob.get("installed") or secret_blob.get("web") or secret_blob
        token_data.setdefault("client_id", inner.get("client_id"))
        token_data.setdefault("client_secret", inner.get("client_secret"))
        token_data.setdefault("token_uri", inner.get("token_uri", "https://oauth2.googleapis.com/token"))

    # Normalize raw Google OAuth Playground JSON seamlessly (access_token -> token, scope -> scopes)
    if "token" not in token_data and "access_token" in token_data:
        token_data["token"] = token_data["access_token"]
    if "scopes" not in token_data and isinstance(token_data.get("scope"), str):
        token_data["scopes"] = [s for s in token_data["scope"].split() if s]
    token_data.setdefault("token_uri", "https://oauth2.googleapis.com/token")

    oauth_creds = Credentials.from_authorized_user_info(token_data)
    if not oauth_creds.valid:
        if not oauth_creds.refresh_token:
            raise RuntimeError(
                "YouTube OAuth token is expired and has no refresh_token. "
                "Re-authorize once with access_type=offline and ensure Google Cloud OAuth app is 'In Production'."
            )
        oauth_creds.refresh(Request())
    return oauth_creds


def _build_resumable_http(oauth_creds: Credentials) -> google_auth_httplib2.AuthorizedHttp:
    """Create an AuthorizedHttp client that NEVER mis-handles YouTube's HTTP 308 Resume Incomplete."""
    http = httplib2.Http(timeout=180)
    if hasattr(http, "redirect_codes") and 308 in http.redirect_codes:
        http.redirect_codes = set(http.redirect_codes) - {308}
    return google_auth_httplib2.AuthorizedHttp(oauth_creds, http=http)


def build_youtube_clients(creds: ApiCredentials, *, local_token_path: Path | None = None) -> tuple[Any, Any]:
    """Return authenticated (youtube_v3, youtube_analytics_v2) service clients."""
    oauth_creds = load_youtube_credentials(creds, local_token_path=local_token_path)
    authed_http_data = _build_resumable_http(oauth_creds)
    authed_http_analytics = _build_resumable_http(oauth_creds)
    youtube = build("youtube", "v3", http=authed_http_data, cache_discovery=False)
    analytics = build("youtubeAnalytics", "v2", http=authed_http_analytics, cache_discovery=False)
    return youtube, analytics


def verify_channel_ownership(youtube: Any, expected_channel_id: str = EXPECTED_CHANNEL_ID) -> dict[str, Any]:
    """Strictly verify OAuth credentials belong to the locked CampUncharted channel."""
    resp = youtube.channels().list(part="snippet,statistics,contentDetails", mine=True).execute()
    items = resp.get("items") or []
    if not items:
        raise RuntimeError("Authenticated Google account has no YouTube channel.")
    channel = items[0]
    actual_id = str(channel.get("id", ""))
    title = str(channel.get("snippet", {}).get("title", ""))
    if expected_channel_id and actual_id != expected_channel_id:
        raise RuntimeError(
            f"CHANNEL SAFETY LOCK: Authenticated channel '{title}' ({actual_id}) "
            f"does NOT match expected CampUncharted channel ({expected_channel_id}). Aborting."
        )
    return {
        "channel_id": actual_id,
        "title": title,
        "uploads_playlist": channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads", ""),
        "subscribers": int(channel.get("statistics", {}).get("subscriberCount", 0) or 0),
        "views": int(channel.get("statistics", {}).get("viewCount", 0) or 0),
        "video_count": int(channel.get("statistics", {}).get("videoCount", 0) or 0),
    }


def check_remote_duplicate(youtube: Any, uploads_playlist_id: str, title: str, *, limit: int = 40) -> str | None:
    """Return existing video_id if a video with essentially the same title already exists on the channel."""
    if not uploads_playlist_id:
        return None
    target = normalize_key(title)
    if not target:
        return None
    try:
        resp = (
            youtube.playlistItems()
            .list(part="snippet", playlistId=uploads_playlist_id, maxResults=min(50, limit))
            .execute()
        )
        for item in resp.get("items", []):
            snip = item.get("snippet", {})
            existing_title = normalize_key(snip.get("title", ""))
            if existing_title == target:
                return str(snip.get("resourceId", {}).get("videoId", ""))
    except HttpError:
        return None
    return None


def reserve_publish_slot(
    store: StateStore,
    *,
    format_type: str = "short",
    publish_now: bool = False,
    now_utc: datetime | None = None,
) -> str | None:
    """Choose the next free peak-traffic UTC publishAt ISO timestamp (or None for immediate public)."""
    if publish_now:
        return None

    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    min_ready = now + timedelta(minutes=45)
    existing = store.get_publish_schedule()
    taken_iso = {str(s.get("publish_at", "")) for s in existing if s.get("format") == format_type}
    slot_times = DOC_UTC_SLOTS if format_type == "documentary" else SHORTS_UTC_SLOTS

    for day_offset in range(0, 21):
        day = (now + timedelta(days=day_offset)).date()
        for hour, minute in slot_times:
            candidate = datetime(day.year, day.month, day.day, hour, minute, 0, tzinfo=timezone.utc)
            if candidate < min_ready:
                continue
            iso_val = candidate.strftime("%Y-%m-%dT%H:%M:%SZ")
            if iso_val not in taken_iso:
                return iso_val

    fallback = (now + timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    return fallback.strftime("%Y-%m-%dT%H:%M:%SZ")


def record_reserved_slot(store: StateStore, *, publish_at: str | None, format_type: str, video_id: str, title: str) -> None:
    if not publish_at:
        return
    slots = store.get_publish_schedule()
    slots.append(
        {
            "publish_at": publish_at,
            "format": format_type,
            "video_id": video_id,
            "title": title,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    store.save_publish_schedule(slots)


def upload_video_resumable(
    youtube: Any,
    video_path: Path,
    *,
    title: str,
    description: str,
    tags: list[str],
    category_id: str,
    publish_at: str | None = None,
    default_language: str = "en",
) -> str:
    """Upload a video cleanly using 8 MiB chunks with HTTP-308 protection and transient retries."""
    status_body: dict[str, Any] = {
        "privacyStatus": "private" if publish_at else "public",
        "selfDeclaredMadeForKids": False,
        "embeddable": True,
        "publicStatsViewable": True,
    }
    if publish_at:
        status_body["publishAt"] = publish_at

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": [t[:30] for t in tags[:12]],
            "categoryId": str(category_id),
            "defaultLanguage": default_language,
            "defaultAudioLanguage": default_language,
        },
        "status": status_body,
    }

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        chunksize=8 * 1024 * 1024,
        resumable=True,
    )
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    retry = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"[YOUTUBE] Upload progress: {int(status.progress() * 100)}%", flush=True)
        except HttpError as err:
            code = getattr(err.resp, "status", 0)
            if code in RETRIABLE_STATUS_CODES and retry < 5:
                retry += 1
                wait = min(45, 2**retry)
                print(f"[YOUTUBE] Transient HTTP {code}; retrying in {wait}s ({retry}/5)", flush=True)
                time.sleep(wait)
                continue
            raise
        except OSError as err:
            if retry < 5:
                retry += 1
                wait = min(45, 2**retry)
                print(f"[YOUTUBE] Network glitch ({err}); retrying in {wait}s ({retry}/5)", flush=True)
                time.sleep(wait)
                continue
            raise

    video_id = str(response.get("id", "")).strip()
    if not video_id:
        raise RuntimeError(f"YouTube upload returned no video ID: {response}")
    return video_id


def set_custom_thumbnail(youtube: Any, video_id: str, thumb_jpg: Path) -> bool:
    """Attach a custom JPEG thumbnail, handling Shorts API restrictions gracefully."""
    if not thumb_jpg.is_file():
        return False
    try:
        media = MediaFileUpload(str(thumb_jpg), mimetype="image/jpeg")
        youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        print(f"[YOUTUBE] Custom thumbnail attached to {video_id}", flush=True)
        return True
    except HttpError as err:
        print(f"[YOUTUBE] Custom thumbnail skipped for {video_id} (HTTP {getattr(err.resp, 'status', '?')})", flush=True)
        return False


def post_engagement_comment(youtube: Any, video_id: str, comment_text: str) -> bool:
    """Post an engaging starter question comment on a public video."""
    text = " ".join(str(comment_text or "").split()).strip()
    if not text:
        return False
    try:
        youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {"snippet": {"textOriginal": text}},
                }
            },
        ).execute()
        print(f"[YOUTUBE] Starter engagement comment posted on {video_id}", flush=True)
        return True
    except HttpError as err:
        print(f"[YOUTUBE] Comment skipped on {video_id} (HTTP {getattr(err.resp, 'status', '?')})", flush=True)
        return False


def add_to_documentary_playlist(
    youtube: Any,
    video_id: str,
    *,
    playlist_title: str = "Nature Documentaries | CampUncharted",
) -> str | None:
    """Ensure the channel's Documentary playlist exists and add the video to it."""
    try:
        resp = youtube.playlists().list(part="snippet", mine=True, maxResults=50).execute()
        playlist_id = None
        for item in resp.get("items", []):
            if normalize_key(item.get("snippet", {}).get("title", "")) == normalize_key(playlist_title):
                playlist_id = item["id"]
                break
        if not playlist_id:
            created = (
                youtube.playlists()
                .insert(
                    part="snippet,status",
                    body={
                        "snippet": {
                            "title": playlist_title,
                            "description": "Full-length cinematic nature & wilderness documentaries from CampUncharted.",
                        },
                        "status": {"privacyStatus": "public"},
                    },
                )
                .execute()
            )
            playlist_id = created.get("id")
        if playlist_id:
            youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id},
                    }
                },
            ).execute()
            print(f"[YOUTUBE] Added {video_id} to playlist '{playlist_title}'", flush=True)
        return playlist_id
    except HttpError as err:
        print(f"[YOUTUBE] Playlist update skipped (HTTP {getattr(err.resp, 'status', '?')})", flush=True)
        return None


def enqueue_video_for_post_processing(
    store: StateStore,
    *,
    video_id: str,
    title: str,
    description: str,
    cues: list[tuple[float, float, str]],
    pinned_comment: str = "",
    comment_posted: bool = False,
    publish_at: str | None = None,
) -> None:
    """Queue a newly uploaded video for quota-safe multi-language captions, localizations, and delayed comment."""
    queue = store.get_subtitle_queue()
    videos = queue.setdefault("videos", {})
    videos[video_id] = {
        "title": title,
        "description": description[:2500],
        "cues": [[round(s, 3), round(e, 3), t] for s, e, t in cues],
        "done_langs": [],
        "localized_langs": [],
        "pinned_comment": pinned_comment,
        "comment_posted": bool(comment_posted),
        "publish_at": publish_at or "",
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }
    # Keep recent 60 videos in queue to keep Git state light
    if len(videos) > 60:
        oldest = sorted(videos.items(), key=lambda kv: kv[1].get("queued_at", ""))[: len(videos) - 60]
        for old_id, _ in oldest:
            videos.pop(old_id, None)
    store.save_subtitle_queue(queue)


def process_subtitle_queue(
    youtube: Any,
    store: StateStore,
    creds: ApiCredentials,
    work_dir: Path,
    *,
    daily_budget: int = DAILY_LANGUAGE_QUOTA_BUDGET,
) -> dict[str, int]:
    """Upload captions + translated localizations + pending starter comments within a strict daily quota cap."""
    queue = store.get_subtitle_queue()
    today = datetime.now(timezone.utc).date().isoformat()
    quota_map = queue.setdefault("quota", {})
    used_today = int(quota_map.get(today, 0) or 0)
    # Prune old dates from quota ledger
    for old_day in [d for d in list(quota_map.keys()) if d != today]:
        quota_map.pop(old_day, None)

    videos = queue.setdefault("videos", {})
    captions_uploaded = 0
    localizations_updated = 0
    comments_posted = 0
    srt_dir = work_dir / "srt_upload"
    srt_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    # Newest videos first so fresh uploads immediately get English & top global subtitles
    ordered = sorted(videos.items(), key=lambda kv: kv[1].get("queued_at", ""), reverse=True)

    for video_id, entry in ordered:
        # 1. Post delayed starter engagement comment once a scheduled video has gone public
        if not entry.get("comment_posted") and entry.get("pinned_comment"):
            pub_raw = str(entry.get("publish_at", "")).strip()
            ready_for_comment = True
            if pub_raw:
                try:
                    pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                    ready_for_comment = now >= pub_dt + timedelta(minutes=2)
                except ValueError:
                    ready_for_comment = True
            if ready_for_comment and post_engagement_comment(youtube, video_id, entry["pinned_comment"]):
                entry["comment_posted"] = True
                comments_posted += 1

        # 2. Multi-language captions & title/description localizations within quota budget
        done_langs = set(entry.get("done_langs") or [])
        localized_langs = set(entry.get("localized_langs") or [])
        raw_cues = entry.get("cues") or []
        if not raw_cues:
            continue
        cues = [(float(c[0]), float(c[1]), str(c[2])) for c in raw_cues]

        pending_localizations: dict[str, dict[str, str]] = {}
        for lang_code, lang_name in SUBTITLE_LANGUAGES.items():
            if lang_code in done_langs:
                continue
            if used_today + CAPTION_INSERT_QUOTA_COST > daily_budget:
                break

            srt_path = srt_dir / f"{video_id}_{lang_code}.srt"
            if lang_code == "en":
                write_srt(cues, srt_path)
            else:
                if not creds.gemini_api_key:
                    continue
                try:
                    translated = translate_subtitles_and_metadata(
                        title=entry["title"],
                        description=entry["description"],
                        lines=[t for _, _, t in cues],
                        target_lang=lang_code,
                        api_key=creds.gemini_api_key,
                    )
                    trans_cues = [
                        (s, e, new_t) for (s, e, _), new_t in zip(cues, translated.translated_lines)
                    ]
                    write_srt(trans_cues, srt_path)
                    if lang_code not in localized_langs:
                        pending_localizations[lang_code] = {
                            "title": translated.localized_title,
                            "description": translated.localized_description,
                        }
                except Exception as err:
                    print(f"[I18N] Translation failed ({video_id} -> {lang_code}): {err}", flush=True)
                    continue

            try:
                media = MediaFileUpload(str(srt_path), mimetype="application/x-subrip")
                youtube.captions().insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "videoId": video_id,
                            "language": lang_code,
                            "name": lang_name,
                            "isDraft": False,
                        }
                    },
                    media_body=media,
                ).execute()
                done_langs.add(lang_code)
                used_today += CAPTION_INSERT_QUOTA_COST
                captions_uploaded += 1
                print(f"[I18N] Uploaded {lang_code} caption for {video_id} (daily quota used: {used_today}/{daily_budget})", flush=True)
            except HttpError as err:
                status = getattr(err.resp, "status", 0)
                if status == 409:
                    # Caption track already exists
                    done_langs.add(lang_code)
                elif status in {403, 429}:
                    print(f"[I18N] Quota/permission stop on caption upload (HTTP {status})", flush=True)
                    used_today = daily_budget
                    break
                else:
                    print(f"[I18N] Caption upload error ({video_id}/{lang_code}): HTTP {status}", flush=True)

        # Apply localized titles & descriptions in a single videos.update call (only 50 units!)
        if pending_localizations and used_today + LOCALIZATION_UPDATE_QUOTA_COST <= daily_budget:
            try:
                vid_resp = youtube.videos().list(part="snippet,localizations", id=video_id).execute()
                items = vid_resp.get("items") or []
                if items:
                    existing_loc = dict(items[0].get("localizations") or {})
                    existing_loc.update(pending_localizations)
                    snippet = items[0].get("snippet", {})
                    snippet.setdefault("defaultLanguage", "en")
                    youtube.videos().update(
                        part="snippet,localizations",
                        body={"id": video_id, "snippet": snippet, "localizations": existing_loc},
                    ).execute()
                    localized_langs.update(pending_localizations.keys())
                    used_today += LOCALIZATION_UPDATE_QUOTA_COST
                    localizations_updated += len(pending_localizations)
                    print(f"[I18N] Updated localizations {sorted(pending_localizations)} for {video_id}", flush=True)
            except HttpError as err:
                print(f"[I18N] Localization update skipped for {video_id} (HTTP {getattr(err.resp, 'status', '?')})", flush=True)

        entry["done_langs"] = sorted(done_langs)
        entry["localized_langs"] = sorted(localized_langs)
        if used_today + CAPTION_INSERT_QUOTA_COST > daily_budget:
            break

    quota_map[today] = used_today
    store.save_subtitle_queue(queue)
    return {
        "captions_uploaded": captions_uploaded,
        "localizations_updated": localizations_updated,
        "comments_posted": comments_posted,
        "quota_used_today": used_today,
    }
