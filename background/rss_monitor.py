"""Multi-source RSS daily digest and on-demand fetch."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, time, timedelta
from typing import Any, Callable, Awaitable, Optional

from services.rss_feeds import fetch_feeds, resolve_feeds, parse_date

SendFunc = Callable[[str], Awaitable[None]]

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
PREFETCH_BUFFER = timedelta(hours=1)

_monitor: Optional["RSSMonitor"] = None


def rank_papers(entries: list[dict], n: int) -> list[dict]:
    """Select top n entries, round-robin across sources for diversity."""
    if len(entries) <= n:
        return entries

    # Group by source
    by_source: dict[str, list[dict]] = {}
    for e in entries:
        src = e.get("source_category", "unknown")
        by_source.setdefault(src, []).append(e)

    # Round-robin pick from each source
    picked: list[dict] = []
    sources = list(by_source.keys())
    idx = {s: 0 for s in sources}
    while len(picked) < n and sources:
        exhausted = []
        for s in sources:
            if len(picked) >= n:
                break
            if idx[s] < len(by_source[s]):
                picked.append(by_source[s][idx[s]])
                idx[s] += 1
            else:
                exhausted.append(s)
        for s in exhausted:
            sources.remove(s)

    return picked


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
        self._load_config()

    def set_send(self, send: SendFunc):
        self._send = send

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

                feeds = resolve_feeds(user_prefs=self._user_prefs)
                results = await fetch_feeds(feeds)
                all_entries = _flatten_entries(results)
                top = rank_papers(all_entries, 10)

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
        top = rank_papers(all_entries, top_n)

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


def _flatten_entries(results: list[dict]) -> list[dict]:
    """Flatten feed results into a single sorted list of entries."""
    entries = []
    for r in results:
        entries.extend(r.get("entries", []))
    entries.sort(key=lambda e: parse_date(e.get("published", "")), reverse=True)
    return entries


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
