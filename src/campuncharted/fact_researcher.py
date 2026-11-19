"""Bounded, verified fact researcher using Wikipedia API & official park sources."""

from __future__ import annotations

import hashlib
import html
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from .config import USER_AGENT
from .state_store import atomic_write_json, read_json


def _strip_html(markup: str) -> str:
    body = re.sub(r"<(script|style|nav|header|footer|aside)\b[^>]*>.*?</\1>", " ", markup, flags=re.S | re.I)
    paragraphs = re.findall(r"<p\b[^>]*>(.*?)</p>", body, flags=re.S | re.I)
    cleaned = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", p)).split()) for p in paragraphs]
    cleaned = [p for p in cleaned if len(p) >= 60]
    return "\n".join(cleaned[:18])[:7000]


def fetch_wikipedia_extract(title: str, *, session: requests.Session | None = None) -> dict[str, str] | None:
    """Fetch clean plaintext summary + full lead/sections from English Wikipedia API."""
    client = session or requests.Session()
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title.replace("_", " "),
        "redirects": 1,
        "prop": "extracts",
        "explaintext": 1,
        "exchars": 7500,
        "format": "json",
        "formatversion": 2,
    }
    try:
        resp = client.get(api, params=params, headers={"User-Agent": USER_AGENT}, timeout=(5, 12))
        if resp.status_code != 200:
            return None
        pages = resp.json().get("query", {}).get("pages", [])
        for page in pages:
            if page.get("missing") or not page.get("extract"):
                continue
            extract = str(page["extract"]).strip()
            if len(extract) < 200:
                continue
            canonical = "https://en.wikipedia.org/wiki/" + quote(page["title"].replace(" ", "_"), safe="_()")
            return {"url": canonical, "title": page["title"], "text": extract[:7500], "kind": "wikipedia"}
    except (requests.RequestException, ValueError, KeyError):
        return None
    return None


def fetch_official_page(url: str, *, session: requests.Session | None = None) -> dict[str, str] | None:
    """Fetch readable paragraph text from an official park/authority page."""
    client = session or requests.Session()
    try:
        resp = client.get(url, headers={"User-Agent": USER_AGENT}, timeout=(6, 15))
        if resp.status_code != 200:
            return None
        text = _strip_html(resp.text)
        if len(text) < 250:
            return None
        return {"url": url, "title": url, "text": text, "kind": "official"}
    except requests.RequestException:
        return None


def research_topic(
    subject: str,
    *,
    wiki_titles: tuple[str, ...] = (),
    official_urls: tuple[str, ...] = (),
    cache_dir: Path | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, str]]:
    """Collect grounded reference text for a Short or Documentary topic with 24h caching."""
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(f"{subject}|{'|'.join(wiki_titles)}|{'|'.join(official_urls)}".encode()).hexdigest()
        cache_path = cache_dir / f"facts_{digest[:16]}.json"
        cached = read_json(cache_path, {})
        if cached.get("expires_at", 0) > time.time() and cached.get("sources"):
            return list(cached["sources"])

    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for url in official_urls[:3]:
        if url in seen_urls or "wikipedia.org" in url:
            continue
        item = fetch_official_page(url, session=session)
        if item:
            sources.append(item)
            seen_urls.add(item["url"])

    for title in wiki_titles[:3]:
        item = fetch_wikipedia_extract(title, session=session)
        if item and item["url"] not in seen_urls:
            sources.append(item)
            seen_urls.add(item["url"])

    # If no explicit wiki titles succeeded, search Wikipedia cleanly for the subject
    if not sources:
        client = session or requests.Session()
        try:
            resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": subject,
                    "gsrnamespace": 0,
                    "gsrlimit": 2,
                    "prop": "extracts",
                    "explaintext": 1,
                    "exchars": 5000,
                    "format": "json",
                    "formatversion": 2,
                },
                headers={"User-Agent": USER_AGENT},
                timeout=(5, 12),
            )
            if resp.status_code == 200:
                for page in resp.json().get("query", {}).get("pages", []):
                    extract = str(page.get("extract", "")).strip()
                    if len(extract) >= 250:
                        canonical = "https://en.wikipedia.org/wiki/" + quote(
                            page["title"].replace(" ", "_"), safe="_()"
                        )
                        if canonical not in seen_urls:
                            sources.append(
                                {"url": canonical, "title": page["title"], "text": extract[:5000], "kind": "wikipedia"}
                            )
                            seen_urls.add(canonical)
        except (requests.RequestException, ValueError):
            pass

    if cache_path is not None and sources:
        atomic_write_json(
            cache_path,
            {"subject": subject, "sources": sources, "expires_at": time.time() + 86400},
        )
    return sources
