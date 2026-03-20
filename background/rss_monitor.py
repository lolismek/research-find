"""arXiv RSS daily digest and on-demand fetch."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, time, timedelta
from typing import Any, Callable, Awaitable, Optional

from services.arxiv import fetch_arxiv_rss

SendFunc = Callable[[str], Awaitable[None]]

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
PREFETCH_BUFFER = timedelta(hours=1)

_monitor: Optional["RSSMonitor"] = None


def rank_papers(papers: list, n: int) -> list:
    """Select top n papers. Placeholder for future scoring logic."""
    return papers[:n]


def _seconds_until(target: time) -> float:
    now = datetime.now().astimezone()
    t = now.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return (t - now).total_seconds()


class RSSMonitor:
    """Daily arXiv digest and on-demand fetch."""

    def __init__(self):
        self._send: Optional[SendFunc] = None
        self._daily_task: Optional[asyncio.Task] = None
        self._notification_time: time = time(9, 0)
        self._load_config()

    def set_send(self, send: SendFunc):
        self._send = send

    def _load_config(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            self._notification_time = time(
                cfg.get("notification_hour", 9),
                cfg.get("notification_minute", 0),
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    def _save_config(self):
        cfg = {
            "notification_hour": self._notification_time.hour,
            "notification_minute": self._notification_time.minute,
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)

    def get_notification_time(self) -> time:
        return self._notification_time

    def set_notification_time(self, hour: int, minute: int = 0):
        self._notification_time = time(hour, minute)
        self._save_config()
        # Restart daily loop with new time
        self.start_daily()

    def start_daily(self):
        """Start (or restart) the daily digest loop."""
        if self._daily_task and not self._daily_task.done():
            self._daily_task.cancel()
        self._daily_task = asyncio.create_task(self._daily_loop())
        t = self._notification_time
        print(f"[rss] Daily arXiv digest scheduled at {t.hour:02d}:{t.minute:02d}")

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

                papers = await fetch_arxiv_rss(None)
                top = rank_papers(papers, 10)

                # Sleep remaining buffer until notification time
                remaining = _seconds_until(t)
                if remaining > 0:
                    await asyncio.sleep(remaining)

                if top and self._send:
                    await self._send(self._format_digest(top, len(papers)))
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

    async def fetch_on_demand(
        self, category: str | None = None, top_n: int = 5
    ) -> dict[str, Any]:
        """One-shot fetch, rank, and return results. Does not push to chat —
        the LLM formats and presents the results itself."""
        papers = await fetch_arxiv_rss(category)
        top = rank_papers(papers, top_n)

        label = category or "all arXiv"
        return {
            "category": label,
            "total_fetched": len(papers),
            "returned": len(top),
            "papers": [
                {
                    "title": p.title,
                    "authors": [a.name for a in p.authors[:3]],
                    "arxiv_id": p.arxiv_id,
                    "url": p.url,
                }
                for p in top
            ],
        }

    @staticmethod
    def _format_digest(papers: list, total: int, label: str = "all arXiv") -> str:
        lines = [f"**[arXiv daily: {label}]** {total} papers found. Top {len(papers)}:\n"]
        for p in papers:
            authors = ", ".join(a.name for a in p.authors[:2])
            if authors:
                lines.append(f"- **{p.title}** — {authors}")
            else:
                lines.append(f"- **{p.title}**")
            if p.url:
                lines.append(f"  {p.url}")
        return "\n".join(lines)


def get_monitor() -> RSSMonitor:
    global _monitor
    if _monitor is None:
        _monitor = RSSMonitor()
    return _monitor
