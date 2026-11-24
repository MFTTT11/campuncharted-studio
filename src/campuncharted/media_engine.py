"""Multi-provider stock video engine (Pexels, Pixabay, Wikimedia Commons) with FFprobe QC."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import ApiCredentials, USER_AGENT
from .state_store import StateStore


@dataclass
class MediaAsset:
    asset_key: str
    provider: str
    asset_id: str
    download_url: str
    page_url: str
    title: str
    author: str
    license_name: str
    width: int
    height: int
    duration: float
    local_path: Path | None = None

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


def sanitize_query(query: str, max_len: int = 85) -> str:
    """Clean query strings so Pixabay (100-char limit) never returns HTTP 400."""
    cleaned = re.sub(r"[^A-Za-z0-9\s-]", " ", str(query or ""))
    words = cleaned.split()
    out: list[str] = []
    for w in words:
        candidate = " ".join(out + [w])
        if len(candidate) > max_len:
            break
        out.append(w)
    return " ".join(out).strip() or "wilderness nature camping"


def probe_video(path: Path) -> dict[str, Any] | None:
    """Inspect a downloaded video file with ffprobe; return dimensions & duration or None."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration,avg_frame_rate:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0:
            return None
        info = json.loads(proc.stdout)
        streams = info.get("streams") or []
        if not streams:
            return None
        s0 = streams[0]
        width = int(s0.get("width") or 0)
        height = int(s0.get("height") or 0)
        dur_raw = s0.get("duration") or info.get("format", {}).get("duration") or 0
        duration = float(dur_raw)
        if width < 640 or height < 480 or duration < 2.5:
            return None
        return {"width": width, "height": height, "duration": duration}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _handle_rate_limit(store: StateStore, provider: str, resp: requests.Response) -> None:
    status = resp.status_code
    retry_after = str(resp.headers.get("Retry-After", "")).strip()
    reset_hdr = str(resp.headers.get("X-Ratelimit-Reset", "")).strip()
    delay = 1800.0 if status in {401, 403, 429} else 300.0
    if retry_after.isdigit():
        delay = float(retry_after)
    elif reset_hdr.replace(".", "", 1).isdigit():
        candidate = float(reset_hdr) - time.time()
        if 30 <= candidate <= 86400:
            delay = candidate
    store.set_provider_cooldown(provider, status, delay, reason=f"HTTP {status}")
    print(f"[MEDIA] {provider} returned HTTP {status}; cooled down for {int(delay)}s", flush=True)


def search_pexels(
    query: str,
    *,
    orientation: str,
    api_key: str,
    store: StateStore,
    per_page: int = 20,
    session: requests.Session | None = None,
) -> list[MediaAsset]:
    """Search Pexels with explicit portrait/landscape orientation (huge Shorts upgrade!)."""
    if not api_key or store.is_provider_cooled_down("Pexels"):
        return []
    client = session or requests.Session()
    clean_q = sanitize_query(query, max_len=80)
    try:
        resp = client.get(
            "https://api.pexels.com/videos/search",
            params={"query": clean_q, "orientation": orientation, "per_page": per_page, "size": "medium"},
            headers={"Authorization": api_key, "User-Agent": USER_AGENT},
            timeout=(5, 15),
        )
        if resp.status_code != 200:
            _handle_rate_limit(store, "Pexels", resp)
            return []
        remaining = resp.headers.get("X-Ratelimit-Remaining")
        if remaining and remaining.isdigit() and int(remaining) <= 3:
            _handle_rate_limit(store, "Pexels", resp)
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    assets: list[MediaAsset] = []
    for video in data.get("videos", []):
        dur = float(video.get("duration") or 0)
        if dur < 3.5:
            continue
        files = [
            f
            for f in video.get("video_files", [])
            if f.get("link") and f.get("file_type", "video/mp4") == "video/mp4" and int(f.get("width") or 0) >= 720
        ]
        if not files:
            continue

        # Pick sharpest 1080p-class file without wasting bandwidth on massive 4K raw files unless needed
        def file_rank(f: dict[str, Any]) -> tuple[int, int]:
            w, h = int(f.get("width") or 0), int(f.get("height") or 0)
            matches_orient = 0 if ((orientation == "portrait" and h >= w) or (orientation == "landscape" and w >= h)) else 1
            target_px = 1080 * 1920
            return (matches_orient, abs(w * h - target_px))

        best = min(files, key=file_rank)
        vid_id = str(video["id"])
        page = str(video.get("url", ""))
        assets.append(
            MediaAsset(
                asset_key=f"pex_{vid_id}",
                provider="Pexels",
                asset_id=vid_id,
                download_url=best["link"],
                page_url=page,
                title=page.rstrip("/").split("/")[-1],
                author=str(video.get("user", {}).get("name", "Pexels contributor")),
                license_name="Pexels License",
                width=int(best.get("width") or 0),
                height=int(best.get("height") or 0),
                duration=dur,
            )
        )
    return assets


def search_pixabay(
    query: str,
    *,
    orientation: str,
    api_key: str,
    store: StateStore,
    per_page: int = 20,
    session: requests.Session | None = None,
) -> list[MediaAsset]:
    """Search Pixabay Videos with sanitized queries (fixes v1 HTTP 400 bug!)."""
    if not api_key or store.is_provider_cooled_down("Pixabay"):
        return []
    client = session or requests.Session()
    # Pixabay strictly enforces <= 100 chars and URL-safe keywords
    clean_q = sanitize_query(query, max_len=70)
    try:
        resp = client.get(
            "https://pixabay.com/api/videos/",
            params={
                "key": api_key,
                "q": clean_q,
                "video_type": "film",
                "per_page": max(3, min(200, per_page)),
                "safesearch": "true",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=(5, 15),
        )
        if resp.status_code != 200:
            _handle_rate_limit(store, "Pixabay", resp)
            return []
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    assets: list[MediaAsset] = []
    for hit in data.get("hits", []):
        dur = float(hit.get("duration") or 0)
        if dur < 3.5:
            continue
        variants = [
            v
            for v in hit.get("videos", {}).values()
            if isinstance(v, dict) and v.get("url") and int(v.get("width") or 0) >= 720
        ]
        if not variants:
            continue
        best = max(variants, key=lambda v: (int(v.get("width") or 0) <= 1920, int(v.get("width") or 0) * int(v.get("height") or 0)))
        vid_id = str(hit["id"])
        assets.append(
            MediaAsset(
                asset_key=f"pix_{vid_id}",
                provider="Pixabay",
                asset_id=vid_id,
                download_url=best["url"],
                page_url=str(hit.get("pageURL", "")),
                title=str(hit.get("tags", "")),
                author=str(hit.get("user", "Pixabay contributor")),
                license_name="Pixabay Content License",
                width=int(best.get("width") or 0),
                height=int(best.get("height") or 0),
                duration=dur,
            )
        )
    return assets


def search_wikimedia_commons(
    query: str,
    *,
    category: str = "",
    limit: int = 20,
    session: requests.Session | None = None,
) -> list[MediaAsset]:
    """Search Wikimedia Commons for CC0 / Public Domain / CC-BY HD videos (great for Documentaries!)."""
    client = session or requests.Session()
    api = "https://commons.wikimedia.org/w/api.php"
    allowed_licenses = {"CC0", "Public domain", "CC BY 4.0", "CC BY 3.0", "CC BY 2.0", "PD-USGov-NPS", "PD-USGov"}
    queries: list[dict[str, Any]] = []
    if category:
        queries.append(
            {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": category,
                "gcmnamespace": 6,
                "gcmlimit": min(50, limit * 2),
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata|size",
                "format": "json",
            }
        )
    clean_q = sanitize_query(query, max_len=60)
    queries.append(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f'filetype:video "{clean_q}"',
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata|size",
            "format": "json",
        }
    )

    assets: list[MediaAsset] = []
    seen: set[str] = set()
    for params in queries:
        try:
            resp = client.get(api, params=params, headers={"User-Agent": USER_AGENT}, timeout=(6, 15))
            if resp.status_code != 200:
                continue
            pages = resp.json().get("query", {}).get("pages", {}).values()
        except (requests.RequestException, ValueError):
            continue
        for page in pages:
            title = str(page.get("title", ""))
            if not title or title in seen or re.search(r"interview|speech|ceremony|president|press", title, re.I):
                continue
            info = (page.get("imageinfo") or [{}])[0]
            if not str(info.get("mime", "")).startswith("video/"):
                continue
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            size = int(info.get("size") or 0)
            if width < 1280 or height < 720 or size <= 0 or size > 120_000_000:
                continue
            meta = info.get("extmetadata", {})
            lic = re.sub(r"<[^>]+>", "", str(meta.get("LicenseShortName", {}).get("value", ""))).strip()
            if lic not in allowed_licenses:
                continue
            url = str(info.get("url", ""))
            if not url:
                continue
            seen.add(title)
            author = re.sub(r"<[^>]+>", "", str(meta.get("Artist", {}).get("value", "Wikimedia contributor"))).strip()[:120]
            title_digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
            assets.append(
                MediaAsset(
                    asset_key=f"wmc_{title_digest}",
                    provider="Wikimedia Commons",
                    asset_id=title,
                    download_url=url,
                    page_url=str(info.get("descriptionurl", "")),
                    title=title,
                    author=author,
                    license_name=lic,
                    width=width,
                    height=height,
                    duration=float(info.get("duration") or 15.0),
                )
            )
    return assets


def download_asset(asset: MediaAsset, dest_path: Path, *, session: requests.Session | None = None) -> bool:
    """Stream-download a video asset and validate it with ffprobe."""
    client = session or requests.Session()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with client.get(asset.download_url, headers={"User-Agent": USER_AGENT}, timeout=(10, 90), stream=True) as resp:
            if resp.status_code != 200:
                return False
            ctype = resp.headers.get("content-type", "").lower()
            if "text/html" in ctype or "application/json" in ctype:
                return False
            total = 0
            with tmp_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > 130_000_000:
                        return False
                    handle.write(chunk)
        if total < 50_000:
            return False
        probed = probe_video(tmp_path)
        if not probed:
            return False
        tmp_path.replace(dest_path)
        asset.local_path = dest_path
        asset.width = probed["width"]
        asset.height = probed["height"]
        asset.duration = probed["duration"]
        return True
    except (requests.RequestException, OSError):
        return False
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def fetch_clips_for_queries(
    queries: list[str],
    *,
    orientation: str,
    target_clips: int,
    work_dir: Path,
    creds: ApiCredentials,
    store: StateStore,
    fallback_queries: tuple[str, ...] = (),
    commons_category: str = "",
) -> list[MediaAsset]:
    """Fetch a diverse, deduplicated set of playable clips across providers."""
    used_ids = store.get_used_media_ids()
    selected: list[MediaAsset] = []
    seen_keys: set[str] = set()
    session = requests.Session()

    all_queries = list(dict.fromkeys([sanitize_query(q) for q in queries if q] + [sanitize_query(q) for q in fallback_queries if q]))
    if orientation == "portrait":
        all_queries.extend(["camping tent mountains nature", "wilderness forest hiking sunset", "mountain river landscape"])
    else:
        all_queries.extend(["mountain wilderness aerial landscape", "forest river nature sunrise", "national park scenic vista"])

    # Gather candidate pool across queries while conserving API calls
    pool: list[MediaAsset] = []
    for idx, query in enumerate(all_queries):
        if len(pool) >= target_clips * 3:
            break
        batch: list[MediaAsset] = []
        batch.extend(search_pexels(query, orientation=orientation, api_key=creds.pexels_api_key, store=store, per_page=15, session=session))
        if len(batch) < 4:
            batch.extend(search_pixabay(query, orientation=orientation, api_key=creds.pixabay_api_key, store=store, per_page=15, session=session))
        if orientation == "landscape" and (idx == 0 or len(batch) < 3):
            batch.extend(
                search_wikimedia_commons(
                    query,
                    category=commons_category if idx == 0 else "",
                    limit=15,
                    session=session,
                )
            )
        for item in batch:
            if item.asset_key in seen_keys or item.asset_key in used_ids:
                continue
            seen_keys.add(item.asset_key)
            pool.append(item)

    # Prefer native orientation first, then crisp resolution & good clip length
    def asset_priority(a: MediaAsset) -> tuple[int, int, float]:
        orient_ok = 0 if ((orientation == "portrait" and a.is_portrait) or (orientation == "landscape" and not a.is_portrait)) else 1
        hd_ok = 0 if max(a.width, a.height) >= 1280 else 1
        return (orient_ok, hd_ok, -min(30.0, a.duration))

    pool.sort(key=asset_priority)

    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for idx, candidate in enumerate(pool):
        if len(selected) >= target_clips:
            break
        ext = ".webm" if candidate.download_url.lower().endswith(".webm") else ".mp4"
        dest = clips_dir / f"clip_{idx:03d}_{candidate.asset_key}{ext}"
        if download_asset(candidate, dest, session=session):
            selected.append(candidate)
            print(
                f"[MEDIA] Ready ({len(selected)}/{target_clips}): {candidate.provider} {candidate.asset_id} "
                f"({candidate.width}x{candidate.height}, {candidate.duration:.1f}s)",
                flush=True,
            )

    return selected
