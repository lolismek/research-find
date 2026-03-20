"""Async PDF downloader for the literature extraction pipeline.

Downloads PDFs from ``paper.pdf_url`` so GROBID can parse them.
Skips papers that already have PMC XML (they don't need GROBID).
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import aiohttp

from src.schemas.paper_schema import Paper

logger = logging.getLogger(__name__)

MAX_PDF_SIZE = 50 * 1024 * 1024  # 50 MB
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=60)


class PDFDownloader:
    """Download PDFs for papers that need GROBID parsing.

    Usage::

        downloader = PDFDownloader()
        pdf_paths = await downloader.download_batch(papers)
        # pdf_paths: {"10.1234/abc": Path("/tmp/.../10.1234_abc.pdf"), ...}
        papers = await pipeline.parse(papers, pdf_paths=pdf_paths)
        downloader.cleanup()
    """

    def __init__(
        self,
        download_dir: Path | None = None,
        max_concurrent: int = 10,
    ) -> None:
        self._tmpdir = None
        if download_dir:
            self.download_dir = download_dir
            self.download_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._tmpdir = tempfile.mkdtemp(prefix="sedona_pdfs_")
            self.download_dir = Path(self._tmpdir)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def download_batch(self, papers: list[Paper]) -> dict[str, Path]:
        """Download PDFs for all eligible papers.

        Returns a mapping of identifier (DOI or PMID) to local PDF path.
        Skips papers that have a PMC ID (PMC XML path is preferred).
        """
        eligible = [
            p for p in papers
            if p.pdf_url and not p.pmc_id
        ]
        if not eligible:
            logger.info("No papers need PDF download (all have PMC IDs or no pdf_url)")
            return {}

        logger.info(f"Downloading PDFs for {len(eligible)}/{len(papers)} papers")

        async with aiohttp.ClientSession(timeout=DOWNLOAD_TIMEOUT) as session:
            tasks = [self._download_one(session, paper) for paper in eligible]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        pdf_paths: dict[str, Path] = {}
        succeeded = 0
        for paper, result in zip(eligible, results):
            if isinstance(result, Exception):
                logger.warning(f"PDF download failed for {paper.doi or paper.pmid}: {result}")
                continue
            if result is None:
                continue
            key = paper.doi or paper.pmid
            if key:
                pdf_paths[key] = result
                succeeded += 1

        logger.info(f"PDF download complete: {succeeded}/{len(eligible)} succeeded")
        return pdf_paths

    async def _download_one(self, session: aiohttp.ClientSession, paper: Paper) -> Path | None:
        """Download a single PDF with semaphore-limited concurrency."""
        async with self._semaphore:
            url = paper.pdf_url
            identifier = paper.doi or paper.pmid or "unknown"
            safe_name = identifier.replace("/", "_").replace(":", "_")
            dest = self.download_dir / f"{safe_name}.pdf"

            if dest.exists():
                logger.debug(f"Already downloaded: {dest}")
                return dest

            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.debug(f"HTTP {resp.status} for {url}")
                        return None

                    content_type = resp.headers.get("Content-Type", "")
                    if "pdf" not in content_type and "octet-stream" not in content_type:
                        logger.debug(f"Unexpected content-type '{content_type}' for {url}")
                        return None

                    content_length = resp.content_length
                    if content_length and content_length > MAX_PDF_SIZE:
                        logger.warning(f"PDF too large ({content_length} bytes) for {identifier}")
                        return None

                    # Stream download with size check
                    total = 0
                    with open(dest, "wb") as f:
                        async for chunk in resp.content.iter_chunked(64 * 1024):
                            total += len(chunk)
                            if total > MAX_PDF_SIZE:
                                logger.warning(f"PDF exceeded {MAX_PDF_SIZE} bytes during download: {identifier}")
                                dest.unlink(missing_ok=True)
                                return None
                            f.write(chunk)

                    logger.debug(f"Downloaded {identifier} ({total} bytes) -> {dest}")
                    return dest

            except asyncio.TimeoutError:
                logger.warning(f"Timeout downloading PDF for {identifier}")
                dest.unlink(missing_ok=True)
                return None
            except aiohttp.ClientError as e:
                logger.warning(f"HTTP error downloading PDF for {identifier}: {e}")
                dest.unlink(missing_ok=True)
                return None

    def cleanup(self) -> None:
        """Remove downloaded PDFs and temp directory."""
        if self._tmpdir:
            import shutil
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            logger.debug(f"Cleaned up PDF temp dir: {self._tmpdir}")
