"""Multi-source RSS daily digest and on-demand fetch."""

from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import datetime, time, timedelta
from typing import Any, Callable, Awaitable, Optional

from tqdm import tqdm

from services.rss_feeds import fetch_feeds, resolve_feeds, parse_date
from services.interest_profile import generate_user_interest_blurb

SendFunc = Callable[[str], Awaitable[None]]

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
PREFETCH_BUFFER = timedelta(hours=1)

_monitor: Optional["RSSMonitor"] = None


def _build_entry_text(entry: dict) -> str:
    """Build text to embed for an RSS entry: title + summary."""
    title = entry.get("title", "")
    summary = entry.get("summary", "")
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Abstract: {summary}")
    return "\n".join(parts)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def rank_papers(
    entries: list[dict], n: int, user_phone: str | None = None,
) -> list[dict]:
    """Rank RSS entries by cosine similarity to user's interest embedding.

    Falls back to recency-only if no user embedding is available.
    """
    from services.neo4j_store import get_user_interest_embedding
    from services.embeddings import embed_texts

    if not entries:
        return []

    # Get user interest embedding
    user_emb = None
    if user_phone:
        user_emb = await get_user_interest_embedding(user_phone)

    if not user_emb:
        print("[rank] No user interest embedding found, returning most recent entries")
        return entries[:n]

    # Build texts for all entries
    texts = [_build_entry_text(e) for e in entries]
    print(f"[rank] Embedding {len(texts)} papers...")

    # Embed in batches of 100 with progress bar
    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []
    pbar = tqdm(total=len(texts), desc="[rank] Embedding", unit="paper")
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_embs = await embed_texts(batch)
        all_embeddings.extend(batch_embs)
        pbar.update(len(batch))
    pbar.close()

    # Score by cosine similarity and return top n
    scored = []
    for entry, emb in zip(entries, all_embeddings):
        sim = _cosine(user_emb, emb)
        scored.append((sim, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"[rank] Top score: {scored[0][0]:.4f}, Bottom score: {scored[-1][0]:.4f}")
    for sim, entry in scored[:n]:
        print(f"[rank]   {sim:.4f} — {entry.get('title', '?')[:80]}")

    return [entry for _, entry in scored[:n]]


def _seconds_until(target: time) -> float:
    now = datetime.now().astimezone()
    t = now.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return (t - now).total_seconds()


class RSSMonitor:
    """Multi-source RSS digest and on-demand fetch."""

    def __init__(self):
        self._send: Optional[SendFunc] = None
        self._daily_task: Optional[asyncio.Task] = None
        self._notification_time: time = time(9, 0)
        self._user_prefs: dict = {}
        self._user_phone: Optional[str] = None
        self._load_config()

    def set_send(self, send: SendFunc):
        self._send = send

    def set_user_phone(self, phone: str):
        self._user_phone = phone

    # ---- config persistence ------------------------------------------------

    def _load_config(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            self._notification_time = time(
                cfg.get("notification_hour", 9),
                cfg.get("notification_minute", 0),
            )
            self._user_prefs = {
                "rss_categories": cfg.get("rss_categories", []),
                "medrxiv_specialties": cfg.get("medrxiv_specialties", []),
                "biorxiv_specialties": cfg.get("biorxiv_specialties", []),
                "arxiv_categories": cfg.get("arxiv_categories", []),
            }
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    def _save_config(self):
        cfg = {
            "notification_hour": self._notification_time.hour,
            "notification_minute": self._notification_time.minute,
            "rss_categories": self._user_prefs.get("rss_categories", []),
            "medrxiv_specialties": self._user_prefs.get("medrxiv_specialties", []),
            "biorxiv_specialties": self._user_prefs.get("biorxiv_specialties", []),
            "arxiv_categories": self._user_prefs.get("arxiv_categories", []),
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)

    # ---- notification time -------------------------------------------------

    def get_notification_time(self) -> time:
        return self._notification_time

    def set_notification_time(self, hour: int, minute: int = 0):
        self._notification_time = time(hour, minute)
        self._save_config()
        self.start_daily()

    # ---- feed preferences --------------------------------------------------

    def set_feed_preferences(
        self,
        rss_categories: list[str] | None = None,
        medrxiv_specialties: list[str] | None = None,
        biorxiv_specialties: list[str] | None = None,
        arxiv_categories: list[str] | None = None,
    ):
        if rss_categories is not None:
            self._user_prefs["rss_categories"] = rss_categories
        if medrxiv_specialties is not None:
            self._user_prefs["medrxiv_specialties"] = medrxiv_specialties
        if biorxiv_specialties is not None:
            self._user_prefs["biorxiv_specialties"] = biorxiv_specialties
        if arxiv_categories is not None:
            self._user_prefs["arxiv_categories"] = arxiv_categories
        self._save_config()

    def get_feed_preferences(self) -> dict:
        return dict(self._user_prefs)

    # ---- daily loop --------------------------------------------------------

    def start_daily(self):
        """Start (or restart) the daily digest loop."""
        if self._daily_task and not self._daily_task.done():
            self._daily_task.cancel()
        self._daily_task = asyncio.create_task(self._daily_loop())
        t = self._notification_time
        print(f"[rss] Daily digest scheduled at {t.hour:02d}:{t.minute:02d}")

    async def _daily_loop(self):
        """Fetch papers before notification time, send digest at the scheduled hour."""
        while True:
            try:
                t = self._notification_time
                prefetch_time = (
                    datetime.combine(datetime.today(), t) - PREFETCH_BUFFER
                ).time()
                sleep_secs = _seconds_until(prefetch_time)
                await asyncio.sleep(sleep_secs)

                if self._user_phone:
                    try:
                        await generate_user_interest_blurb(self._user_phone)
                    except Exception as e:
                        print(f"[rss] Interest blurb generation failed: {e}")

                feeds = resolve_feeds(user_prefs=self._user_prefs)
                results = await fetch_feeds(feeds)
                all_entries = _flatten_entries(results)
                top = await rank_papers(all_entries, 10, user_phone=self._user_phone)

                remaining = _seconds_until(t)
                if remaining > 0:
                    await asyncio.sleep(remaining)

                if top and self._send:
                    await self._send(self._format_digest(top, len(all_entries)))
                elif top:
                    print(f"[rss] {len(top)} papers ready but no chat connected")

            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[rss] Daily loop error: {e}")
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    return

    # ---- on-demand fetch ---------------------------------------------------

    async def fetch_on_demand(
        self,
        source: str | None = None,
        category: str | None = None,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """One-shot fetch, rank, and return results."""
        if self._user_phone:
            try:
                await generate_user_interest_blurb(self._user_phone)
            except Exception as e:
                print(f"[rss] Interest blurb generation failed: {e}")

        feeds = resolve_feeds(source=source, category=category, user_prefs=self._user_prefs)
        if not feeds:
            return {
                "source": source or "all",
                "category": category,
                "total_fetched": 0,
                "returned": 0,
                "papers": [],
                "error": f"No feeds found for source={source!r}, category={category!r}",
            }

        results = await fetch_feeds(feeds)
        all_entries = _flatten_entries(results)
        top = await rank_papers(all_entries, top_n, user_phone=self._user_phone)

        label_parts = []
        if source:
            label_parts.append(source)
        if category:
            label_parts.append(category)
        label = " / ".join(label_parts) or "all configured"

        return {
            "source": label,
            "category": category,
            "total_fetched": len(all_entries),
            "returned": len(top),
            "papers": [
                {
                    "title": e["title"],
                    "authors": _format_entry_authors(e.get("authors", [])),
                    "link": e.get("link", ""),
                    "published": e.get("published", ""),
                    "source_category": e.get("source_category", ""),
                    "arxiv_id": e.get("arxiv_id"),
                    "doi": e.get("doi"),
                    "summary": (e.get("summary") or "")[:300],
                }
                for e in top
            ],
        }

    # ---- formatting --------------------------------------------------------

    @staticmethod
    def _format_digest(entries: list[dict], total: int) -> str:
        lines = [f"**[Daily digest]** {total} papers found. Top {len(entries)}:\n"]
        for e in entries:
            authors = _format_entry_authors(e.get("authors", []))
            author_str = ", ".join(authors[:2])
            source = e.get("source_category", "")
            title_line = f"- **{e['title']}**"
            if author_str:
                title_line += f" -- {author_str}"
            if source:
                title_line += f" [{source}]"
            lines.append(title_line)
            link = e.get("link", "")
            if link:
                lines.append(f"  {link}")
        return "\n".join(lines)


def _dedup_key(entry: dict) -> str:
    """Return a key for deduplication: prefer DOI, then arXiv ID, then normalized title."""
    doi = (entry.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    arxiv = (entry.get("arxiv_id") or "").strip().lower()
    if arxiv:
        return f"arxiv:{arxiv}"
    title = (entry.get("title") or "").strip().lower()
    # collapse whitespace for robustness
    title = " ".join(title.split())
    return f"title:{title}"


def _flatten_entries(results: list[dict]) -> list[dict]:
    """Flatten feed results into a single sorted, deduplicated list of entries."""
    entries = []
    for r in results:
        entries.extend(r.get("entries", []))
    entries.sort(key=lambda e: parse_date(e.get("published", "")), reverse=True)

    seen: set[str] = set()
    deduped = []
    for e in entries:
        key = _dedup_key(e)
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    if len(entries) != len(deduped):
        print(f"[rss] Deduplicated {len(entries)} → {len(deduped)} entries")
    return deduped


def _format_entry_authors(authors: Any) -> list[str]:
    """Normalize authors from entry dicts to a list of name strings."""
    if not authors:
        return []
    out = []
    for a in authors[:3]:
        if isinstance(a, dict):
            out.append(a.get("name", "Unknown"))
        elif isinstance(a, str):
            out.append(a)
    return out


def get_monitor() -> RSSMonitor:
    global _monitor
    if _monitor is None:
        _monitor = RSSMonitor()
    return _monitor
