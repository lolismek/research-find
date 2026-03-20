"""arXiv RSS polling + scheduled notifications."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

from services.arxiv import fetch_arxiv_rss

_monitor: Optional[RSSMonitor] = None


class RSSMonitor:
    """Manages background RSS polling tasks and notification queue."""

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._configs: dict[str, dict] = {}
        self._notifications: list[dict[str, Any]] = []

    def start_monitoring(
        self,
        category: str,
        schedule: str = "immediate",
        top_n: int = 5,
    ):
        """Start monitoring an arXiv category.

        Args:
            category: arXiv category (e.g., "cs.AI")
            schedule: "immediate" or a time like "9am"
            top_n: Number of top papers per notification
        """
        if category in self._tasks and not self._tasks[category].done():
            return  # Already monitoring

        self._configs[category] = {
            "schedule": schedule,
            "top_n": top_n,
        }
        task = asyncio.create_task(self._poll_loop(category))
        self._tasks[category] = task

    def stop_monitoring(self, category: str):
        """Stop monitoring an arXiv category."""
        task = self._tasks.pop(category, None)
        if task and not task.done():
            task.cancel()
        self._configs.pop(category, None)

    def active_categories(self) -> set[str]:
        """Return set of categories being monitored."""
        return {cat for cat, task in self._tasks.items() if not task.done()}

    def get_pending_notifications(self) -> list[dict[str, Any]]:
        """Return and clear pending notifications that are ready."""
        now = datetime.utcnow()
        ready = [n for n in self._notifications if n.get("ready_at", now) <= now]
        self._notifications = [n for n in self._notifications if n not in ready]
        return ready

    async def _poll_loop(self, category: str):
        """Background loop: fetch RSS, queue notifications, sleep 12 hours."""
        seen_ids: set[str] = set()

        while True:
            try:
                papers = await fetch_arxiv_rss(category)
                config = self._configs.get(category, {})
                top_n = config.get("top_n", 5)
                schedule = config.get("schedule", "immediate")

                new_papers = []
                for p in papers:
                    pid = p.arxiv_id or p.title
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        new_papers.append(p)

                if new_papers:
                    top_papers = new_papers[:top_n]
                    notif = {
                        "category": category,
                        "papers": [
                            {
                                "title": p.title,
                                "arxiv_id": p.arxiv_id,
                                "authors": [a.name for a in p.authors[:3]],
                                "abstract": (p.abstract or "")[:200],
                                "url": p.url,
                            }
                            for p in top_papers
                        ],
                        "total_new": len(new_papers),
                        "fetched_at": datetime.utcnow().isoformat(),
                    }

                    if schedule == "immediate":
                        notif["ready_at"] = datetime.utcnow()
                    else:
                        # For simplicity, mark as ready immediately
                        # A full implementation would parse schedule time
                        notif["ready_at"] = datetime.utcnow()

                    self._notifications.append(notif)

            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[RSS Monitor] Error fetching {category}: {e}")

            # Poll every 12 hours (arXiv RSS updates daily)
            try:
                await asyncio.sleep(12 * 3600)
            except asyncio.CancelledError:
                return


def get_monitor() -> RSSMonitor:
    """Get or create the singleton RSS monitor."""
    global _monitor
    if _monitor is None:
        _monitor = RSSMonitor()
    return _monitor
