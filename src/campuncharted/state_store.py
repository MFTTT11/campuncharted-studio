"""Atomic, Git-friendly durable state store for deduplication, scheduling, and insights."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def normalize_key(value: str) -> str:
    """Normalize text for case-insensitive, accent-insensitive deduplication."""
    raw = str(value or "").casefold().replace("ı", "i").replace("ø", "o")
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9]+", " ", stripped).strip()
    return re.sub(r"\bturkiye\b", "turkey", cleaned)


def content_hash(parts: list[str]) -> str:
    normalized = [normalize_key(p) for p in parts if str(p or "").strip()]
    return hashlib.sha256("|".join(normalized).encode("utf-8")).hexdigest()


CONCEPT_STOPWORDS = frozenset(
    {
        "a", "above", "actually", "after", "an", "and", "are", "around", "at", "be", "before",
        "beside", "best", "between", "by", "call", "camp", "camper", "campers", "camping", "campsite",
        "can", "cause", "causes", "causing", "cold", "danger", "dangerous", "deadly", "do", "does",
        "during", "earth", "fatal", "for", "from", "gear", "guide", "guides", "hack", "hacks",
        "hazard", "hidden", "hike", "hikers", "hiking", "how", "in", "inside", "instant", "into",
        "is", "it", "level", "make", "makes", "mistake", "mistakes", "most", "mountain", "mountains",
        "must", "national", "natural", "naturally", "nature", "never", "night", "of", "on", "or",
        "outdoor", "outdoors", "over", "park", "pitch", "pitching", "place", "places", "rapid",
        "real", "really", "risk", "rule", "rules", "safe", "safety", "science", "secret", "secrets",
        "short", "shorts", "silent", "simple", "spot", "spots", "stay", "stays", "still", "survival",
        "survive", "tent", "that", "the", "this", "tip", "tips", "to", "trail", "trend", "triggers",
        "truth", "under", "using", "weather", "what", "when", "where", "why", "wild", "wilderness",
        "wise", "with", "without", "you", "your",
    }
)


def concept_tokens(*parts: str) -> set[str]:
    """Extract distinctive normalized content keywords for near-duplicate topic detection."""
    tokens: set[str] = set()
    for part in parts:
        for tok in normalize_key(part).split():
            if len(tok) >= 3 and tok not in CONCEPT_STOPWORDS and not tok.isdigit():
                tokens.add(tok)
    return tokens


def is_near_duplicate_concept(candidate: set[str], existing_sets: list[set[str]]) -> bool:
    """Return True if candidate shares >=2 distinctive keywords (>=28% of smaller set) or >=45% Jaccard overlap."""
    if not candidate:
        return False
    for other in existing_sets:
        if not other:
            continue
        overlap = candidate & other
        if not overlap:
            continue
        if len(overlap) >= 3:
            return True
        if len(overlap) >= 2 and (len(overlap) / min(len(candidate), len(other))) >= 0.28:
            return True
        union = candidate | other
        if len(overlap) / len(union) >= 0.45:
            return True
    return False


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class StateStore:
    """Persistent repository-backed memory for CampUncharted Studio."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.state_dir / name

    # --- Uploaded locations & topics ---
    def get_uploaded_locations(self) -> list[str]:
        data = read_json(self._path("uploaded_locations.json"), {"locations": []})
        return list(data.get("locations", []))

    def is_location_uploaded(self, location: str) -> bool:
        target = normalize_key(location.split(",", 1)[0])
        full = normalize_key(location)
        if not target:
            return False
        for existing in self.get_uploaded_locations():
            base = normalize_key(existing.split("|", 1)[0].split(",", 1)[0])
            if target == base or full == normalize_key(existing):
                return True
        return False

    def record_uploaded_location(self, location: str) -> None:
        items = self.get_uploaded_locations()
        if not self.is_location_uploaded(location):
            items.append(location.strip())
            atomic_write_json(
                self._path("uploaded_locations.json"),
                {"updated_at": datetime.now(timezone.utc).isoformat(), "locations": items},
            )

    # --- Topic history (for series/topic IDs beyond pure locations) ---
    def get_topic_history(self) -> list[dict[str, Any]]:
        data = read_json(self._path("topic_history.json"), {"topics": []})
        return list(data.get("topics", []))

    def is_topic_used(self, topic_id: str) -> bool:
        key = normalize_key(topic_id)
        return any(normalize_key(row.get("topic_id", "")) == key for row in self.get_topic_history())

    def record_topic(
        self,
        *,
        topic_id: str,
        series: str,
        title: str,
        video_id: str = "",
        duration_seconds: float = 0.0,
        format_type: str = "short",
    ) -> None:
        rows = self.get_topic_history()
        rows.append(
            {
                "topic_id": topic_id,
                "series": series,
                "title": title,
                "video_id": video_id,
                "duration_seconds": round(float(duration_seconds), 1),
                "format": format_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(
            self._path("topic_history.json"),
            {"updated_at": datetime.now(timezone.utc).isoformat(), "topics": rows},
        )

    def recent_series(self, limit: int = 6) -> list[str]:
        shorts = [r.get("series", "") for r in self.get_topic_history() if r.get("format") == "short"]
        return [s for s in shorts[-limit:] if s]

    # --- Used media IDs ---
    def get_used_media_ids(self) -> set[str]:
        data = read_json(self._path("used_media_ids.json"), {"media_ids": []})
        return set(str(x) for x in data.get("media_ids", []) if x)

    def record_used_media_ids(self, new_ids: set[str] | list[str]) -> None:
        merged = sorted(self.get_used_media_ids() | {str(x) for x in new_ids if x})
        atomic_write_json(
            self._path("used_media_ids.json"),
            {"updated_at": datetime.now(timezone.utc).isoformat(), "count": len(merged), "media_ids": merged},
        )

    # --- Content hashes ---
    def get_content_hashes(self) -> set[str]:
        data = read_json(self._path("content_hashes.json"), {"hashes": []})
        return set(str(x) for x in data.get("hashes", []) if x)

    def record_content_hash(self, digest: str) -> None:
        merged = sorted(self.get_content_hashes() | {digest})
        atomic_write_json(
            self._path("content_hashes.json"),
            {"updated_at": datetime.now(timezone.utc).isoformat(), "hashes": merged},
        )

    # --- Documentary archive ---
    def get_documentary_archive(self) -> dict[str, Any]:
        return dict(read_json(self._path("documentary_archive.json"), {}))

    def is_documentary_done(self, place_slug: str) -> bool:
        archive = self.get_documentary_archive()
        key = normalize_key(place_slug).replace(" ", "_")
        return key in archive or any(normalize_key(k) == normalize_key(place_slug) for k in archive)

    def record_documentary(self, place: str, video_id: str, title: str, duration_seconds: float) -> None:
        archive = self.get_documentary_archive()
        slug = normalize_key(place).replace(" ", "_")
        archive[slug] = {
            "place": place,
            "video_id": video_id,
            "title": title,
            "duration_seconds": round(float(duration_seconds), 1),
            "published_at": datetime.now(timezone.utc).isoformat(),
            "status": "UPLOADED" if video_id else "DRY_RUN",
        }
        atomic_write_json(self._path("documentary_archive.json"), archive)

    # --- Publish schedule ---
    def get_publish_schedule(self) -> list[dict[str, Any]]:
        data = read_json(self._path("publish_schedule.json"), {"slots": []})
        return list(data.get("slots", []))

    def save_publish_schedule(self, slots: list[dict[str, Any]]) -> None:
        # Keep the last 120 slots to stay compact in Git.
        trimmed = slots[-120:]
        atomic_write_json(
            self._path("publish_schedule.json"),
            {"updated_at": datetime.now(timezone.utc).isoformat(), "slots": trimmed},
        )

    # --- Provider cooldowns ---
    def get_provider_cooldowns(self) -> dict[str, Any]:
        return dict(read_json(self._path("provider_cooldowns.json"), {}))

    def is_provider_cooled_down(self, provider: str) -> bool:
        row = self.get_provider_cooldowns().get(provider, {})
        return float(row.get("retry_at", 0) or 0) > time.time()

    def set_provider_cooldown(self, provider: str, http_status: int, retry_seconds: float, reason: str = "") -> None:
        data = self.get_provider_cooldowns()
        data[provider] = {
            "http_status": int(http_status),
            "retry_at": round(time.time() + max(60.0, min(86400.0, float(retry_seconds))), 1),
            "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self._path("provider_cooldowns.json"), data)

    # --- Subtitle queue & daily language quota ---
    def get_subtitle_queue(self) -> dict[str, Any]:
        return dict(read_json(self._path("subtitle_queue.json"), {"quota": {}, "videos": {}}))

    def save_subtitle_queue(self, queue_data: dict[str, Any]) -> None:
        queue_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self._path("subtitle_queue.json"), queue_data)

    # --- Channel analytics & dynamic insights ---
    def get_channel_insights(self) -> dict[str, Any]:
        return dict(read_json(self._path("channel_insights.json"), {}))

    def save_channel_insights(self, insights: dict[str, Any]) -> None:
        insights["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self._path("channel_insights.json"), insights)

    def get_analytics_history(self) -> dict[str, Any]:
        return dict(read_json(self._path("analytics_history.json"), {"snapshots": []}))

    def append_analytics_snapshot(self, snapshot: dict[str, Any]) -> None:
        history = self.get_analytics_history()
        snapshots = list(history.get("snapshots", []))
        snap_date = str(snapshot.get("date", "")).strip()
        if snap_date:
            snapshots = [row for row in snapshots if str(row.get("date", "")).strip() != snap_date]
        snapshots.append(snapshot)
        # Keep 90 days of daily snapshots
        history["snapshots"] = snapshots[-90:]
        history["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(self._path("analytics_history.json"), history)
