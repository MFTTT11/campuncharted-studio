"""High-CTR Pro Thumbnail Studio: Gemini/Imagen AI + 4K Stock Photo Engine + broadcast Pillow/Numpy compositor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import io
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import requests
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat

from .audio_engine import probe_audio_duration
from .config import USER_AGENT
from .media_engine import sanitize_query
from .subtitle_engine import detect_bold_font_path

# Preferred Gemini native image & Imagen models (auto-discovered dynamically via models.list)
PREFERRED_IMAGE_MODELS = (
    "gemini-3.1-flash-image-preview",
    "gemini-3.0-flash-image",
    "gemini-2.5-flash-image",
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-3.0-generate-002",
)

DANGER_HOOK_WORDS = frozenset(
    {
        "NEVER", "DEADLY", "KILLER", "FATAL", "FREEZE", "DANGER", "STOP", "WARNING",
        "TRAP", "FAIL", "FAILING", "FORBIDDEN", "BANNED", "WORST", "MISTAKE", "DONT",
        "KILL", "FLOOD", "HYPOTHERMIA", "DEATH", "SILENT", "USELESS", "TOXIC", "BROKEN",
    }
)

# Always reject text overlays, indoor/urban shots, posed portraits, selfies, and tourist crowds
HARD_PHOTO_REJECT_RE = re.compile(
    r"\b("
    r"portrait|selfie|face|faces|tourist|tourists|smiling|smile|posing|pose|model|"
    r"crowd|group|family|couple|friends|wedding|bride|groom|baby|kid|kids|child|children|"
    r"city|town|street|building|buildings|hotel|resort|house|apartment|architecture|"
    r"car|cars|vehicle|truck|van|bus|motorcycle|traffic|highway|"
    r"sign|signs|text|logo|watermark|banner|poster|phone|laptop|screen|"
    r"indoor|interior|room|office|studio|dog|dogs|pet|pets|cat"
    r")\b",
    re.IGNORECASE,
)

# Strongly prefer pure untouched wilderness/nature scenes (no people, boats, roads, or clothes)
SOFT_PHOTO_REJECT_RE = re.compile(
    r"\b("
    r"woman|women|man|men|people|person|persons|girl|girls|boy|boys|lady|gentleman|guy|"
    r"hiker|hikers|backpacker|backpackers|traveler|travelers|camper|campers|"
    r"canoe|canoes|kayak|kayaks|boat|boats|yacht|ship|ferry|paddle|rowing|raft|dock|pier|"
    r"road|bridge|train|bike|bicycle|village|"
    r"hands|hand|holding|standing|sitting|walking|wearing|jacket|coat|hat|glasses|sunglasses|horse"
    r")\b",
    re.IGNORECASE,
)

GENERIC_SERIES_LABELS = frozenset(
    {
        "CAMPUNCHARTED",
        "DOCUMENTARY",
        "CAMP MISTAKES",
        "CAMPING MISTAKE",
        "GEAR AND HACKS",
        "EXTREME WEATHER",
        "EXTREME WEATHER NATURE",
        "TURKEY AND EURASIA",
        "TURKEY AND EURASIA HIDDEN",
        "WILD STAYS",
        "WILDERNESS TRUTH",
        "SURVIVAL RULE",
        "WINTER SURVIVAL RULE",
        "OUTDOOR SAFETY",
    }
)

PHOTO_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "by",
        "why", "how", "what", "when", "where", "who", "which", "this", "that", "these", "those",
        "your", "you", "their", "our", "its", "can", "could", "will", "would", "should", "must",
        "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "dont", "cant",
        "has", "have", "had", "into", "from", "with", "without", "about", "inside", "outside",
        "under", "over", "above", "below", "between", "during", "before", "after", "through",
        "never", "always", "still", "actually", "really", "just", "only", "even", "most", "more",
        "turn", "turns", "turning", "make", "makes", "making", "get", "gets", "keep", "keeps",
        "fail", "fails", "failing", "freeze", "freezes", "freezing", "look", "looking", "see",
        "one", "two", "three", "five", "ten", "foot", "feet", "dollar", "dollars", "minutes",
        "seconds", "hours", "days", "year", "years", "wildest", "hidden", "secret", "secrets",
        "truth", "shocking", "surprising", "deadly", "fatal", "silent", "worst", "best", "rule",
        "rules", "mistake", "mistakes", "hack", "hacks", "tip", "tips", "reason", "reasons",
        "physics", "science", "behind", "explained", "guide", "campers", "hikers", "backpackers",
        "people", "humans", "experienced", "seasoned", "wise", "ordinary", "single", "completely",
    }
)

SCENE_PRIORITY_NOUNS = frozenset(
    {
        "turquoise", "glacier", "glacial", "lake", "lakes", "mountain", "mountains", "alpine",
        "waterfall", "waterfalls", "canyon", "desert", "dunes", "volcano", "forest", "pine",
        "snow", "winter", "aurora", "campfire", "fire", "tent", "bear", "grizzly", "river",
        "cliff", "cliffs", "fjord", "valley", "ridge", "storm", "lightning", "fog", "mist",
        "ice", "cave", "island", "ocean", "coast", "redwood", "sequoia", "arch", "geyser",
    }
)


@dataclass
class StockPhotoCandidate:
    provider: str
    photo_id: str
    download_url: str
    page_url: str
    width: int
    height: int
    description: str
    purity_tier: int  # 0 = zero people/boats/roads, 1 = outdoor fallback
    query_rank: int
    popularity: int = 0


def _is_danger_hook(hook_text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", str(hook_text or "").upper())
    return any(tok in DANGER_HOOK_WORDS for tok in tokens)


def list_candidate_image_models(client: Any) -> list[tuple[str, str]]:
    """Discover available image-generation models on this API key.

    Returns ordered list of `(mode, model_name)` where `mode` is `"gemini_image"` or `"imagen"`.
    """
    try:
        available = list(client.models.list())
    except Exception:
        return [
            ("imagen" if m.startswith("imagen") else "gemini_image", m)
            for m in PREFERRED_IMAGE_MODELS
        ]

    discovered: dict[str, str] = {}
    for model in available:
        name = str(getattr(model, "name", "")).removeprefix("models/")
        lower = name.lower()
        actions = [str(a) for a in (getattr(model, "supported_actions", None) or [])]
        if "imagen" in lower and (not actions or "predict" in actions or "generateImages" in actions):
            discovered[name] = "imagen"
        elif (
            "gemini" in lower
            and "image" in lower
            and "embedding" not in lower
            and not lower.startswith("gemini-1.")
            and (not actions or "generateContent" in actions)
        ):
            discovered[name] = "gemini_image"

    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(m: str, mode: str) -> None:
        if m and m not in seen:
            seen.add(m)
            ordered.append((mode, m))

    # 1. Exact or prefix matches against our preferred newest-first list
    for pref in PREFERRED_IMAGE_MODELS:
        for name, mode in discovered.items():
            if name == pref or pref in name:
                _add(name, mode)

    # 2. Any newer discovered Gemini image models (reverse sorted so 4.x/3.x come first)
    for name in sorted(discovered, reverse=True):
        if discovered[name] == "gemini_image":
            _add(name, "gemini_image")

    # 3. Any discovered Imagen models
    for name in sorted(discovered, reverse=True):
        if discovered[name] == "imagen":
            _add(name, "imagen")

    if not ordered:
        for pref in PREFERRED_IMAGE_MODELS:
            _add(pref, "imagen" if pref.startswith("imagen") else "gemini_image")
    return ordered


def build_thumbnail_scene_prompt(
    *,
    title: str,
    subject: str = "",
    hook_text: str = "",
    format_type: str = "documentary",
    mood: str = "EPIC_DISCOVERY",
) -> str:
    """Build a zero-text, high-CTR cinematic photography prompt engineered for mobile YouTube feeds."""
    is_short = format_type == "short"
    danger = _is_danger_hook(hook_text or title)

    lighting = (
        "dramatic storm-lit twilight with intense warm amber campfire/headlamp highlights against deep cold slate-blue shadows"
        if danger
        else "breathtaking golden-hour cinematic sunlight, crisp atmospheric light rays, vivid natural teal and warm gold contrast"
    )
    composition = (
        "Vertical 9:16 composition. Place the bold, unmistakable main outdoor subject in the upper-center half (top 18% to 55%) "
        "with rich dark atmospheric foreground space in the lower third for clean visual separation."
        if is_short
        else "Widescreen 16:9 Rule-of-Thirds composition. Place the breathtaking illuminated hero subject sharply on the RIGHT 55% "
        "of the frame, and leave clean dark moody natural shadows on the LEFT 40% as negative space."
    )
    topic_desc = f"{title.strip()}. {subject.strip()}".strip(" .")

    return (
        f"Ultra-sharp National Geographic documentary photograph representing: {topic_desc}. "
        f"{composition} "
        f"Lighting & mood ({mood}): {lighting}. "
        "Shot on 35mm full-frame lens, razor-sharp hero focal point, high contrast, vivid realistic colors, "
        "instantly readable even as a tiny mobile thumbnail. "
        "STRICT RULES: ABSOLUTELY NO text, NO letters, NO words, NO numbers, NO signs, NO logos, NO watermarks, NO split-screen borders."
    )


def _validate_image_bytes(raw_bytes: bytes, out_path: Path) -> Path | None:
    """Validate decoded image bytes for dimensions, contrast, and non-blank quality, saving PNG."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            if min(w, h) < 360 or max(w, h) < 640:
                return None
            stat = ImageStat.Stat(rgb.resize((240, 135), Image.Resampling.BILINEAR))
            brightness = sum(stat.mean) / 3.0
            contrast = sum(stat.stddev) / 3.0
            if brightness < 16.0 or brightness > 238.0 or contrast < 12.0:
                return None
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(out_path, "PNG")
            return out_path
    except Exception:
        return None


def _extract_gemini_image_bytes(response: Any) -> bytes | None:
    """Extract raw image bytes from a google-genai generate_content response."""
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if isinstance(data, (bytes, bytearray)) and len(data) > 4096:
                return bytes(data)
    return None


def generate_ai_thumbnail_background(
    out_png: Path,
    *,
    title: str,
    subject: str = "",
    hook_text: str = "",
    format_type: str = "documentary",
    mood: str = "EPIC_DISCOVERY",
    api_key: str = "",
    client: Any | None = None,
) -> Path | None:
    """Generate a bespoke zero-text cinematic thumbnail scene via Gemini/Imagen with graceful fallback."""
    if not api_key and client is None:
        return None

    active_client = client or genai.Client(api_key=api_key, http_options={"timeout": 60000})
    aspect = "9:16" if format_type == "short" else "16:9"
    prompt = build_thumbnail_scene_prompt(
        title=title,
        subject=subject,
        hook_text=hook_text,
        format_type=format_type,
        mood=mood,
    )

    candidates = list_candidate_image_models(active_client)[:4]
    for mode, model_name in candidates:
        try:
            raw_bytes: bytes | None = None
            if mode == "imagen":
                resp = active_client.models.generate_images(
                    model=model_name,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio=aspect,
                        output_mime_type="image/jpeg",
                    ),
                )
                generated = getattr(resp, "generated_images", None) or []
                if generated:
                    img_obj = getattr(generated[0], "image", None)
                    raw_bytes = getattr(img_obj, "image_bytes", None)
            else:
                resp = active_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio=aspect),
                    ),
                )
                raw_bytes = _extract_gemini_image_bytes(resp)

            if raw_bytes:
                saved = _validate_image_bytes(raw_bytes, out_png)
                if saved is not None:
                    print(f"[THUMB] Generated bespoke AI scene via {model_name} ({aspect})", flush=True)
                    return saved
                print(f"[THUMB] AI image from '{model_name}' failed visual QC; trying next model...", flush=True)
        except Exception as err:
            print(f"[THUMB] AI image model '{model_name}' unavailable ({str(err)[:130]}); trying next...", flush=True)
            continue

    print("[THUMB] AI thumbnail background unavailable; trying 4K stock photography next.", flush=True)
    return None


def _normalize_photo_meta_text(raw: str) -> str:
    return re.sub(r"[-_/,.:;]+", " ", str(raw or "")).lower()


def _classify_photo_purity(meta_text: str) -> int | None:
    """Return 0 for pure wilderness, 1 for outdoor fallback, or None for rejected tourist/urban/text shots."""
    normalized = _normalize_photo_meta_text(meta_text)
    if HARD_PHOTO_REJECT_RE.search(normalized):
        return None
    if SOFT_PHOTO_REJECT_RE.search(normalized):
        return 1
    return 0


def _extract_clean_keywords(text: str, *, max_words: int = 4) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{3,}", str(text or "").lower())
    seen: set[str] = set()
    priority: list[str] = []
    regular: list[str] = []
    for tok in tokens:
        if tok in PHOTO_QUERY_STOPWORDS or tok in seen:
            continue
        if SOFT_PHOTO_REJECT_RE.fullmatch(tok) or HARD_PHOTO_REJECT_RE.fullmatch(tok):
            continue
        seen.add(tok)
        if tok in SCENE_PRIORITY_NOUNS:
            priority.append(tok)
        else:
            regular.append(tok)
    return (priority + regular)[:max_words]


def _singularize_scene_noun(word: str) -> str:
    if word in {"lakes", "mountains", "glaciers", "waterfalls", "cliffs", "canyons", "forests", "rivers", "peaks", "trees"}:
        return word[:-1]
    return word


def build_photo_search_queries(
    *,
    title: str,
    subject: str = "",
    place_label: str = "",
    visual_queries: Sequence[str] = (),
    format_type: str = "documentary",
) -> list[str]:
    """Build 2-4 crisp 2-4 word photography queries prioritizing place + scenic nouns (never raw question prose)."""
    queries: list[str] = []
    seen: set[str] = set()

    def _add(raw_q: str, *, max_words: int = 4) -> None:
        cleaned = sanitize_query(raw_q, max_len=55)
        words = [
            w for w in cleaned.split()
            if not HARD_PHOTO_REJECT_RE.fullmatch(w) and not SOFT_PHOTO_REJECT_RE.fullmatch(w)
        ]
        words = words[:max_words]
        if len(words) < 2:
            return
        q = " ".join(words)
        key = q.lower()
        if key not in seen:
            seen.add(key)
            queries.append(q)

    combined_text = f"{title} {subject}"
    scene_nouns: list[str] = []
    for tok in re.findall(r"[A-Za-z]{3,}", combined_text.lower()):
        if tok in SCENE_PRIORITY_NOUNS:
            norm = _singularize_scene_noun(tok)
            if norm not in scene_nouns:
                scene_nouns.append(norm)

    # 1. Explicit non-generic place label (e.g., "Banff National Park" -> "Banff lake turquoise" + "Banff National Park")
    raw_place = str(place_label or "").strip()
    extra_place_q = ""
    if raw_place and raw_place.upper() not in GENERIC_SERIES_LABELS:
        clean_place = sanitize_query(raw_place, max_len=45)
        short_place = re.sub(
            r"\b(national park|national forest|state park|national monument|region)\b",
            "",
            clean_place,
            flags=re.IGNORECASE,
        )
        short_place = " ".join(short_place.split()) or clean_place
        if scene_nouns:
            _add(f"{short_place} {' '.join(scene_nouns[:2])}", max_words=4)
        if visual_queries:
            _add(str(visual_queries[0]), max_words=4)
        if len(clean_place.split()) >= 2:
            _add(clean_place, max_words=4)
        extra_place_q = f"{short_place} mountain landscape" if "mountain" not in short_place.lower() else f"{short_place} landscape"

    # 2. Curated topic/beat visual queries (trimmed to crisp photography keywords)
    for vq in visual_queries or ():
        _add(str(vq), max_words=4)
        if len(queries) >= 4:
            break

    if extra_place_q and len(queries) < 4:
        _add(extra_place_q, max_words=4)

    # 3. Clean visual keywords extracted from title + subject (stripped of "Why/How/Failing/10-Foot" clickbait filler)
    if len(queries) < 4:
        kw = _extract_clean_keywords(combined_text, max_words=4)
        if kw:
            if len(kw) == 1:
                kw.append("wilderness")
            _add(" ".join(kw), max_words=4)

    # 4. Scenic wilderness fallback query (guaranteed high-res hits if niche query returns empty)
    fallback_q = "dramatic wilderness mountain nature" if format_type == "short" else "dramatic mountain wilderness landscape"
    if fallback_q.lower() not in seen:
        queries.append(fallback_q)
    return queries[:5]


def search_pexels_photos(
    query: str,
    *,
    orientation: str,
    api_key: str,
    per_page: int = 15,
    query_rank: int = 0,
    session: Any | None = None,
) -> list[StockPhotoCandidate]:
    """Search Pexels 4K Photography API and filter out portraits, tourists, boats, urban scenes, and text."""
    if not api_key:
        return []
    orient = "portrait" if orientation == "portrait" else "landscape"
    clean_q = sanitize_query(query, max_len=65)
    client = session or requests.Session()
    try:
        resp = client.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": api_key, "User-Agent": USER_AGENT},
            params={"query": clean_q, "orientation": orient, "size": "large", "per_page": per_page},
            timeout=15,
        )
    except Exception:
        return []
    if getattr(resp, "status_code", 0) != 200:
        return []

    try:
        payload = resp.json()
    except Exception:
        return []

    out: list[StockPhotoCandidate] = []
    for photo in payload.get("photos") or []:
        try:
            w = int(photo.get("width") or 0)
            h = int(photo.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if orient == "portrait":
            if h < 1200 or w < 720 or h <= w:
                continue
        else:
            if w < 1200 or h < 675 or w <= h:
                continue

        alt = str(photo.get("alt") or "")
        page_url = str(photo.get("url") or "")
        purity = _classify_photo_purity(f"{alt} {page_url}")
        if purity is None:
            continue

        src = photo.get("src") or {}
        dl_url = str(
            src.get("large2x")
            or src.get(orient)
            or src.get("large")
            or src.get("original")
            or ""
        ).strip()
        if not dl_url:
            continue

        out.append(
            StockPhotoCandidate(
                provider="Pexels",
                photo_id=str(photo.get("id") or ""),
                download_url=dl_url,
                page_url=page_url,
                width=w,
                height=h,
                description=f"{alt} {page_url}".strip(),
                purity_tier=purity,
                query_rank=query_rank,
                popularity=0,
            )
        )
    out.sort(key=lambda c: c.purity_tier)
    return out


def search_pixabay_photos(
    query: str,
    *,
    orientation: str,
    api_key: str,
    per_page: int = 15,
    query_rank: int = 0,
    session: Any | None = None,
) -> list[StockPhotoCandidate]:
    """Search Pixabay Photography API and filter out portraits, tourists, boats, urban scenes, and text."""
    if not api_key:
        return []
    orient = "portrait" if orientation == "portrait" else "landscape"
    clean_q = sanitize_query(query, max_len=65)
    client = session or requests.Session()
    try:
        resp = client.get(
            "https://pixabay.com/api/",
            headers={"User-Agent": USER_AGENT},
            params={
                "key": api_key,
                "q": clean_q,
                "image_type": "photo",
                "orientation": "vertical" if orient == "portrait" else "horizontal",
                "safesearch": "true",
                "order": "popular",
                "min_width": 720 if orient == "portrait" else 1200,
                "min_height": 1200 if orient == "portrait" else 675,
                "per_page": per_page,
            },
            timeout=15,
        )
    except Exception:
        return []
    if getattr(resp, "status_code", 0) != 200:
        return []

    try:
        payload = resp.json()
    except Exception:
        return []

    out: list[StockPhotoCandidate] = []
    for hit in payload.get("hits") or []:
        try:
            w = int(hit.get("imageWidth") or hit.get("webformatWidth") or 0)
            h = int(hit.get("imageHeight") or hit.get("webformatHeight") or 0)
            likes = int(hit.get("likes") or 0)
        except (TypeError, ValueError):
            continue
        if orient == "portrait":
            if h < 1200 or w < 720 or h <= w:
                continue
        else:
            if w < 1200 or h < 675 or w <= h:
                continue

        tags = str(hit.get("tags") or "")
        page_url = str(hit.get("pageURL") or "")
        purity = _classify_photo_purity(f"{tags} {page_url}")
        if purity is None:
            continue

        dl_url = str(hit.get("largeImageURL") or hit.get("fullHDURL") or hit.get("webformatURL") or "").strip()
        if not dl_url:
            continue

        out.append(
            StockPhotoCandidate(
                provider="Pixabay",
                photo_id=str(hit.get("id") or ""),
                download_url=dl_url,
                page_url=page_url,
                width=w,
                height=h,
                description=f"{tags} {page_url}".strip(),
                purity_tier=purity,
                query_rank=query_rank,
                popularity=likes,
            )
        )
    out.sort(key=lambda c: (c.purity_tier, -c.popularity))
    return out


def _score_frame(image_path: Path) -> float:
    """Score frame visual energy using sharpness, saturation, contrast, and balanced brightness."""
    with Image.open(image_path) as raw:
        rgb = raw.convert("RGB").resize((320, 180), Image.Resampling.BILINEAR)
        hsv = rgb.convert("HSV")
        gray = rgb.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)

        rgb_stat = ImageStat.Stat(rgb)
        hsv_stat = ImageStat.Stat(hsv)
        edge_stat = ImageStat.Stat(edges)

        brightness = sum(rgb_stat.mean) / 3.0
        contrast = sum(rgb_stat.stddev) / 3.0
        saturation = hsv_stat.mean[1]
        sharpness = edge_stat.mean[0]

        # Penalize nearly black or overexposed white frames
        brightness_penalty = 0.0
        if brightness < 35:
            brightness_penalty = (35 - brightness) * 1.5
        elif brightness > 220:
            brightness_penalty = (brightness - 220) * 1.5

        return (sharpness * 1.6) + (saturation * 0.55) + (contrast * 0.85) - brightness_penalty


def _region_energy(rgb_crop: Image.Image) -> float:
    hsv = rgb_crop.convert("HSV")
    edges = rgb_crop.convert("L").filter(ImageFilter.FIND_EDGES)
    rgb_stat = ImageStat.Stat(rgb_crop)
    hsv_stat = ImageStat.Stat(hsv)
    edge_stat = ImageStat.Stat(edges)
    return (edge_stat.mean[0] * 1.5) + (hsv_stat.mean[1] * 0.5) + ((sum(rgb_stat.stddev) / 3.0) * 0.8)


def _score_thumbnail_photo(
    image_path: Path,
    *,
    is_short: bool,
    purity_tier: int = 0,
    query_rank: int = 0,
    keyword_hits: int = 0,
) -> float:
    """Score a candidate photograph for CTR visual energy, composition fit, purity, and topic relevance."""
    base = _score_frame(image_path)
    with Image.open(image_path) as raw:
        rgb_full = raw.convert("RGB")
        target = (180, 320) if is_short else (320, 180)
        rgb = rgb_full.resize(target, Image.Resampling.BILINEAR)
        hsv_stat = ImageStat.Stat(rgb.convert("HSV"))
        rgb_stat = ImageStat.Stat(rgb)

        brightness = sum(rgb_stat.mean) / 3.0
        contrast = sum(rgb_stat.stddev) / 3.0
        saturation = hsv_stat.mean[1]

        # Mildly penalize dull/foggy/murky photos so vivid high-CTR scenes win
        dull_penalty = 0.0
        if brightness < 45:
            dull_penalty += (45 - brightness) * 0.6
        if contrast < 28:
            dull_penalty += (28 - contrast) * 0.9
        if saturation < 32:
            dull_penalty += (32 - saturation) * 0.5

        # Composition bonus aligned with our typography overlay zones
        comp_bonus = 0.0
        if is_short:
            # 9:16 hero sits in upper-middle window (15%..58%) above lower-third hook text
            hero = rgb.crop((18, 48, 162, 186))
            hero_energy = _region_energy(hero)
            comp_bonus = min(12.0, hero_energy * 0.08)
        else:
            # 16:9 text sits on left 55%, bright hero shines on right 50%
            left_energy = _region_energy(rgb.crop((0, 0, 160, 180)))
            right_energy = _region_energy(rgb.crop((160, 0, 320, 180)))
            if right_energy >= left_energy * 0.85:
                comp_bonus = min(12.0, (right_energy / max(1.0, left_energy)) * 6.0)
            else:
                comp_bonus = min(6.0, right_energy * 0.05)

    purity_bonus = 18.0 if purity_tier == 0 else 0.0
    rank_bonus = max(0.0, 10.0 - query_rank * 3.0)
    relevance_bonus = min(12.0, keyword_hits * 4.0)
    return base + comp_bonus + purity_bonus + rank_bonus + relevance_bonus - dull_penalty


def _candidate_keyword_hits(candidate: StockPhotoCandidate, topic_tokens: set[str]) -> int:
    if not topic_tokens:
        return 0
    meta_tokens = set(re.findall(r"[a-z]{3,}", _normalize_photo_meta_text(candidate.description)))
    return len(meta_tokens & topic_tokens)


def fetch_best_stock_photo_background(
    work_dir: Path,
    *,
    title: str,
    subject: str = "",
    place_label: str = "",
    visual_queries: Sequence[str] = (),
    format_type: str = "documentary",
    pexels_api_key: str = "",
    pixabay_api_key: str = "",
    max_download: int = 8,
    session: Any | None = None,
) -> Path | None:
    """Search Pexels + Pixabay 4K photography APIs, filter tourists/portraits, and score top candidates."""
    if not pexels_api_key and not pixabay_api_key:
        return None

    is_short = format_type == "short"
    orient = "portrait" if is_short else "landscape"
    queries = build_photo_search_queries(
        title=title,
        subject=subject,
        place_label=place_label,
        visual_queries=visual_queries,
        format_type=format_type,
    )
    if not queries:
        return None

    topic_tokens = set(_extract_clean_keywords(f"{place_label} {title} {subject}", max_words=10))
    client = session or requests.Session()

    all_candidates: list[StockPhotoCandidate] = []
    seen_keys: set[tuple[str, str]] = set()
    primary_queries = queries[:-1] if len(queries) > 1 else queries
    fallback_query = queries[-1] if len(queries) > 1 else None

    def _collect(q_list: Sequence[str], start_rank: int = 0) -> None:
        for offset, q in enumerate(q_list):
            rank = start_rank + offset
            batch: list[StockPhotoCandidate] = []
            if pexels_api_key:
                batch.extend(
                    search_pexels_photos(
                        q,
                        orientation=orient,
                        api_key=pexels_api_key,
                        per_page=15,
                        query_rank=rank,
                        session=client,
                    )
                )
            if pixabay_api_key:
                batch.extend(
                    search_pixabay_photos(
                        q,
                        orientation=orient,
                        api_key=pixabay_api_key,
                        per_page=15,
                        query_rank=rank,
                        session=client,
                    )
                )
            for cand in batch:
                key = (cand.provider, cand.photo_id or cand.download_url)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_candidates.append(cand)
            pure_count = sum(1 for c in all_candidates if c.purity_tier == 0)
            if pure_count >= 12 and offset >= 1:
                break

    _collect(primary_queries, start_rank=0)
    if len(all_candidates) < 2 and fallback_query:
        _collect([fallback_query], start_rank=len(primary_queries))

    if not all_candidates:
        print(f"[THUMB] Stock photo search returned 0 filtered candidates for queries={queries[:3]}", flush=True)
        return None

    # Sort candidates within each provider and interleave Pexels + Pixabay for head-to-head visual scoring
    def _rank_key(c: StockPhotoCandidate) -> tuple[int, int, int, int]:
        return (
            c.purity_tier,
            c.query_rank,
            -_candidate_keyword_hits(c, topic_tokens),
            -c.popularity,
        )

    interleaved: list[StockPhotoCandidate] = []
    for tier in (0, 1):
        pexels_pool = sorted(
            [c for c in all_candidates if c.provider == "Pexels" and c.purity_tier == tier],
            key=_rank_key,
        )
        pixabay_pool = sorted(
            [c for c in all_candidates if c.provider == "Pixabay" and c.purity_tier == tier],
            key=_rank_key,
        )
        for i in range(max(len(pexels_pool), len(pixabay_pool))):
            if i < len(pexels_pool):
                interleaved.append(pexels_pool[i])
            if i < len(pixabay_pool):
                interleaved.append(pixabay_pool[i])

    photos_dir = work_dir / "thumb_photo_candidates"
    photos_dir.mkdir(parents=True, exist_ok=True)

    best_path: Path | None = None
    best_cand: StockPhotoCandidate | None = None
    best_score = float("-inf")
    evaluated = 0

    for idx, cand in enumerate(interleaved):
        if evaluated >= max_download:
            break
        try:
            resp = client.get(cand.download_url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if getattr(resp, "status_code", 0) != 200:
                continue
            raw_bytes = getattr(resp, "content", b"")
            if not raw_bytes or len(raw_bytes) > 12_000_000:
                continue
        except Exception:
            continue

        out_png = photos_dir / f"cand_{idx:02d}_{cand.provider.lower()}_{cand.photo_id or idx}.png"
        saved = _validate_image_bytes(raw_bytes, out_png)
        if saved is None:
            continue

        evaluated += 1
        hits = _candidate_keyword_hits(cand, topic_tokens)
        score = _score_thumbnail_photo(
            saved,
            is_short=is_short,
            purity_tier=cand.purity_tier,
            query_rank=cand.query_rank,
            keyword_hits=hits,
        )
        if score > best_score:
            best_score = score
            best_path = saved
            best_cand = cand

    if best_path is not None and best_cand is not None:
        print(
            f"[THUMB] Selected 4K stock photo: {best_cand.provider} #{best_cand.photo_id} "
            f"({best_cand.width}x{best_cand.height}, tier={best_cand.purity_tier}, score={best_score:.1f}) "
            f"from {evaluated} scored candidates | queries={queries[:3]}",
            flush=True,
        )
        return best_path

    print("[THUMB] Downloaded stock photo candidates failed visual QC; falling back to video frame.", flush=True)
    return None


def extract_best_frame(mp4_path: Path, work_dir: Path, *, sample_count: int = 7) -> Path:
    """Extract evenly spaced frames from MP4 and return the highest visual-energy frame."""
    duration = probe_audio_duration(mp4_path)
    frames_dir = work_dir / "thumb_candidates"
    frames_dir.mkdir(parents=True, exist_ok=True)

    best_path: Path | None = None
    best_score = float("-inf")
    for idx in range(sample_count):
        frac = 0.12 + (0.72 * idx / max(1, sample_count - 1))
        ts = max(0.3, min(duration - 0.3, duration * frac))
        out_png = frames_dir / f"cand_{idx:02d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(mp4_path),
                "-frames:v",
                "1",
                str(out_png),
            ],
            check=True,
            timeout=30,
        )
        if out_png.is_file():
            score = _score_frame(out_png)
            if score > best_score:
                best_score = score
                best_path = out_png

    if not best_path:
        raise RuntimeError(f"Failed to extract thumbnail candidate frames from {mp4_path}")
    return best_path


def resolve_thumbnail_background(
    *,
    work_dir: Path,
    title: str,
    subject: str = "",
    hook_text: str = "",
    format_type: str = "documentary",
    mood: str = "EPIC_DISCOVERY",
    api_key: str = "",
    pexels_api_key: str = "",
    pixabay_api_key: str = "",
    place_label: str = "",
    visual_queries: Sequence[str] = (),
    fallback_video: Path | None = None,
    sample_count: int = 7,
    client: Any | None = None,
    http_session: Any | None = None,
) -> tuple[Path, str]:
    """Return `(background_image_path, source_kind)` with 3-tier priority: AI -> 4K Stock Photo -> Video Frame."""
    ai_path = generate_ai_thumbnail_background(
        work_dir / f"ai_thumb_bg_{format_type}.png",
        title=title,
        subject=subject,
        hook_text=hook_text,
        format_type=format_type,
        mood=mood,
        api_key=api_key,
        client=client,
    )
    if ai_path is not None and ai_path.is_file():
        return ai_path, "ai"

    photo_path = fetch_best_stock_photo_background(
        work_dir,
        title=title,
        subject=subject,
        place_label=place_label,
        visual_queries=visual_queries,
        format_type=format_type,
        pexels_api_key=pexels_api_key,
        pixabay_api_key=pixabay_api_key,
        session=http_session,
    )
    if photo_path is not None and photo_path.is_file():
        return photo_path, "stock_photo"

    if fallback_video is not None and fallback_video.is_file():
        return extract_best_frame(fallback_video, work_dir, sample_count=sample_count), "video_frame"
    raise RuntimeError("Could not obtain thumbnail background (AI and 4K stock photos unavailable and no fallback video provided).")


def _load_truetype(size: int) -> ImageFont.FreeTypeFont:
    font_path = detect_bold_font_path()
    if font_path is None:
        raise RuntimeError("No bold TrueType font installed (install fonts-roboto or fonts-dejavu-core).")
    return ImageFont.truetype(str(font_path), size=size)


def _fit_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale = max(target_w / max(1, src_w), target_h / max(1, src_h))
    new_w = max(target_w, int(round(src_w * scale)))
    new_h = max(target_h, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _split_hook_lines(hook_text: str) -> list[str]:
    words = [w.strip() for w in hook_text.upper().split() if w.strip()]
    if not words:
        return ["WILDERNESS", "TRUTH"]
    if len(words) <= 2:
        return words
    mid = (len(words) + 1) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _apply_cinema_grade_and_scrim(canvas: Image.Image, *, is_short: bool, is_danger: bool) -> Image.Image:
    """Apply vectorized S-curve contrast pop, cinematic vignette, and directional readability scrim."""
    # Boost saturation/clarity first so thumbnails pop off both dark & light YouTube feeds
    graded = ImageEnhance.Color(canvas).enhance(1.24)
    graded = ImageEnhance.Contrast(graded).enhance(1.14)
    graded = ImageEnhance.Sharpness(graded).enhance(1.24)

    arr = np.asarray(graded, dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    # Gentle S-curve micro-contrast (deepens murky shadows slightly while brightening focal highlights)
    arr = np.clip(0.5 + (arr - 0.5) * 1.06 + 0.03 * np.sin(np.pi * (arr * 2.0 - 1.0)), 0.0, 1.0)

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    # Natural radial cinema vignette focusing attention on the hero region
    cx, cy = (0.50, 0.38) if is_short else (0.64, 0.44)
    dist2 = ((xx - cx) / 0.72) ** 2 + ((yy - cy) / 0.72) ** 2
    vignette = 1.0 - np.clip(dist2 - 0.45, 0.0, 1.0) * 0.28
    arr *= vignette[..., None]

    # Directional scrim: protects typography while keeping hero visual bright & vibrant!
    if is_short:
        top_scrim = np.clip((0.18 - yy) / 0.18, 0.0, 1.0) ** 1.5 * 0.52
        bot_scrim = np.clip((yy - 0.50) / 0.50, 0.0, 1.0) ** 1.35 * 0.86
        scrim_alpha = np.maximum(top_scrim, bot_scrim)
    else:
        # 16:9 Documentary: asymmetric left + bottom-left editorial scrim (right hero stays crystal bright!)
        left_scrim = np.clip((0.62 - xx) / 0.62, 0.0, 1.0) ** 1.45 * 0.78
        bot_scrim = np.clip((yy - 0.46) / 0.54, 0.0, 1.0) ** 1.4 * (0.58 * np.clip(1.15 - xx, 0.25, 1.0))
        top_left = np.clip((0.20 - yy) / 0.20, 0.0, 1.0) ** 1.5 * 0.32
        scrim_alpha = np.maximum(np.maximum(left_scrim, bot_scrim), top_left)

    # Tint scrim deep obsidian-navy (or subtly warmer charcoal for danger)
    scrim_rgb = np.array([0.04, 0.03, 0.05] if is_danger else [0.025, 0.04, 0.065], dtype=np.float32)
    blended = arr * (1.0 - scrim_alpha[..., None]) + scrim_rgb[None, None, :] * scrim_alpha[..., None]

    # Subtle ambient accent glow near text anchor for depth
    glow_color = np.array([1.0, 0.27, 0.23] if is_danger else [1.0, 0.82, 0.22], dtype=np.float32)
    gx, gy = (0.50, 0.72) if is_short else (0.14, 0.78)
    gdist2 = ((xx - gx) / 0.34) ** 2 + ((yy - gy) / 0.26) ** 2
    glow = np.exp(-gdist2 * 2.4) * 0.11
    blended = np.clip(blended + glow[..., None] * glow_color[None, None, :], 0.0, 1.0)

    out_rgb = Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return out_rgb.convert("RGBA")


def generate_thumbnail(
    source_frame: Path,
    out_jpg: Path,
    *,
    hook_text: str,
    sub_label: str = "",
    format_type: str = "documentary",
) -> Path:
    """Render a crisp, high-CTR broadcast JPEG thumbnail (<1.9 MB) for Documentary (16:9) or Short (9:16)."""
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    is_short = format_type == "short"
    width, height = (1080, 1920) if is_short else (1280, 720)
    is_danger = _is_danger_hook(hook_text)

    with Image.open(source_frame) as raw:
        canvas = _fit_cover(raw.convert("RGB"), width, height)

    composed = _apply_cinema_grade_and_scrim(canvas, is_short=is_short, is_danger=is_danger)

    # Palette: Danger hooks get Electric Coral-Red + White; Discovery/Hacks get NatGeo Gold + White
    accent = (255, 69, 58, 255) if is_danger else (255, 212, 59, 255)
    white = (255, 255, 255, 255)
    dark_stroke = (8, 10, 14, 255)
    pill_bg = (14, 16, 22, 215)

    # Separate shadow layer so typography floats cleanly with real Gaussian drop-shadow depth
    shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    fg_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(fg_layer)

    # 1. Crisp editorial kicker pill with vector icon (never relies on Unicode font glyph coverage!)
    badge_text = (sub_label.strip() or "CAMPUNCHARTED").upper()[:24]
    badge_font = _load_truetype(34 if is_short else 26)
    bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    pad_x, pad_y = (28, 13) if is_short else (22, 10)
    icon_size = 18 if is_short else 14
    icon_gap = 14 if is_short else 11
    total_pill_w = bw + 2 * pad_x + icon_size + icon_gap
    total_pill_h = bh + 2 * pad_y
    if is_short:
        bx0 = (width - total_pill_w) // 2
        by0 = 132
    else:
        bx0 = 60
        by0 = 46

    pill_box = [bx0, by0, bx0 + total_pill_w, by0 + total_pill_h]
    shadow_draw.rounded_rectangle(
        [pill_box[0], pill_box[1] + 5, pill_box[2], pill_box[3] + 5],
        radius=14,
        fill=(0, 0, 0, 170),
    )
    draw.rounded_rectangle(
        pill_box,
        radius=14,
        fill=pill_bg,
        outline=accent,
        width=3,
    )
    icon_cx = bx0 + pad_x + icon_size // 2
    icon_cy = by0 + total_pill_h // 2
    r = icon_size // 2
    if is_danger:
        draw.polygon(
            [(icon_cx, icon_cy - r), (icon_cx - r, icon_cy + r), (icon_cx + r, icon_cy + r)],
            fill=accent,
        )
    else:
        draw.ellipse([icon_cx - r, icon_cy - r, icon_cx + r, icon_cy + r], fill=accent)

    text_x = bx0 + pad_x + icon_size + icon_gap - bbox[0]
    text_y = by0 + (total_pill_h - bh) // 2 - bbox[1]
    draw.text((text_x, text_y), badge_text, font=badge_font, fill=accent)

    # 2. Massive impact hook typography (tight leading, auto-fitted to safe width)
    lines = _split_hook_lines(hook_text)
    # For 16:9, keep hook text cleanly inside the left ~58% so right focal hero + bottom-right timestamp stay unobstructed!
    max_text_w = int(width * (0.88 if is_short else 0.58))
    font_size = 152 if is_short else 132
    font = _load_truetype(font_size)
    while font_size > 54:
        font = _load_truetype(font_size)
        widest = max((draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in lines), default=0)
        if widest <= max_text_w:
            break
        font_size -= 6

    line_gap = int(font_size * 0.12)
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [b[3] - b[1] for b in bboxes]
    total_h = sum(line_heights) + line_gap * max(0, len(lines) - 1)
    stroke_w = max(6, font_size // 12)

    if is_short:
        start_y = int(height * 0.70) - total_h // 2
        for idx, (line, lb) in enumerate(zip(lines, bboxes)):
            lw = lb[2] - lb[0]
            tx = (width - lw) // 2 - lb[0]
            ty = start_y - lb[1]
            color = accent if idx == len(lines) - 1 else white
            shadow_draw.text(
                (tx, ty + 12),
                line,
                font=font,
                fill=(0, 0, 0, 235),
                stroke_width=stroke_w + 4,
                stroke_fill=(0, 0, 0, 235),
            )
            draw.text(
                (tx, ty),
                line,
                font=font,
                fill=color,
                stroke_width=stroke_w,
                stroke_fill=dark_stroke,
            )
            start_y += line_heights[idx] + line_gap
    else:
        # 16:9 Documentary cover: bold lower-left editorial layout with glowing vertical accent bar
        bar_x = 60
        start_y = height - total_h - 62
        shadow_draw.rounded_rectangle(
            [bar_x - 2, start_y + 4, bar_x + 18, start_y + total_h + 6],
            radius=8,
            fill=(accent[0], accent[1], accent[2], 145),
        )
        draw.rounded_rectangle(
            [bar_x, start_y + 4, bar_x + 15, start_y + total_h - 2],
            radius=7,
            fill=accent,
        )
        tx_base = bar_x + 36
        for idx, (line, lb) in enumerate(zip(lines, bboxes)):
            tx = tx_base - lb[0]
            ty = start_y - lb[1]
            color = accent if idx == len(lines) - 1 else white
            shadow_draw.text(
                (tx + 4, ty + 10),
                line,
                font=font,
                fill=(0, 0, 0, 235),
                stroke_width=stroke_w + 4,
                stroke_fill=(0, 0, 0, 235),
            )
            draw.text(
                (tx, ty),
                line,
                font=font,
                fill=color,
                stroke_width=stroke_w,
                stroke_fill=dark_stroke,
            )
            start_y += line_heights[idx] + line_gap

    # Composite blurred drop-shadow layer then crisp foreground typography
    blurred_shadow = shadow_layer.filter(ImageFilter.GaussianBlur(radius=10 if is_short else 8))
    composed = Image.alpha_composite(composed, blurred_shadow)
    composed = Image.alpha_composite(composed, fg_layer)

    # Save under YouTube's strict 2 MB custom thumbnail limit
    final_rgb = composed.convert("RGB")
    for quality in (92, 86, 78):
        final_rgb.save(out_jpg, "JPEG", quality=quality, optimize=True, progressive=True)
        if out_jpg.stat().st_size <= 1_900_000:
            break
    return out_jpg
