"""Multi-source RSS feed catalog, filtering, and async fetching."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any

import feedparser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# medRxiv specialty extraction
# ---------------------------------------------------------------------------

_MEDRXIV_HTML = """<select name="selectedPage" onchange="changePage(this.form.selectedPage)"><option value="">Select a subject category &nbsp;</option><option value="http://connect.medrxiv.org/medrxiv_xml.php?subject=all">All</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Addiction_Medicine">Addiction Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Allergy_and_Immunology">Allergy and Immunology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Anesthesia">Anesthesia</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Cardiovascular_Medicine">Cardiovascular Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Dentistry_and_Oral_Medicine">Dentistry and Oral Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Dermatology">Dermatology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Emergency_Medicine">Emergency Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=endocrinology">Endocrinology (including Diabetes Mellitus and Metabolic Disease)</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Epidemiology">Epidemiology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Forensic_Medicine">Forensic Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Gastroenterology">Gastroenterology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Genetic_and_Genomic_Medicine">Genetic and Genomic Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Geriatric_Medicine">Geriatric Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Health_Economics">Health Economics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Health_Informatics">Health Informatics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Health_Policy">Health Policy</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Health_Systems_and_Quality_Improvement">Health Systems and Quality Improvement</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Hematology">Hematology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=hivaids">HIV/AIDS</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=infectious_diseases">Infectious Diseases (except HIV/AIDS)</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Intensive_Care_and_Critical_Care_Medicine">Intensive Care and Critical Care Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Medical_Education">Medical Education</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Medical_Ethics">Medical Ethics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Nephrology">Nephrology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Neurology">Neurology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Nursing">Nursing</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Nutrition">Nutrition</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Obstetrics_and_Gynecology">Obstetrics and Gynecology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Occupational_and_Environmental_Health">Occupational and Environmental Health</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Oncology">Oncology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Ophthalmology">Ophthalmology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Orthopedics">Orthopedics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Otolaryngology">Otolaryngology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Pain_Medicine">Pain Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Palliative_Medicine">Palliative Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Pathology">Pathology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Pediatrics">Pediatrics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Pharmacology_and_Therapeutics">Pharmacology and Therapeutics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Primary_Care_Research">Primary Care Research</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Psychiatry_and_Clinical_Psychology">Psychiatry and Clinical Psychology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Public_and_Global_Health">Public and Global Health</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Radiology_and_Imaging">Radiology and Imaging</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Rehabilitation_Medicine_and_Physical_Therapy">Rehabilitation Medicine and Physical Therapy</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Respiratory_Medicine">Respiratory Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Rheumatology">Rheumatology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Sexual_and_Reproductive_Health">Sexual and Reproductive Health</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Sports_Medicine">Sports Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Surgery">Surgery</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Toxicology">Toxicology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Transplantation">Transplantation</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Urology">Urology</option> </select>"""


def _extract_medrxiv_urls(html: str) -> list[str]:
    pattern = r'value="(http://connect\.medrxiv\.org/medrxiv_xml\.php\?subject=[^"]*)"'
    return list(set(url for url in re.findall(pattern, html) if url))


# ---------------------------------------------------------------------------
# Feed catalog
# ---------------------------------------------------------------------------

ALL_AVAILABLE_FEEDS: dict[str, list[str]] = {
    "nature": [
        "https://www.nature.com/nm.rss",
        "https://www.nature.com/nbt.rss",
        "https://www.nature.com/nature.rss",
        "https://www.nature.com/ng.rss",
        "https://www.nature.com/ni.rss",
        "https://www.nature.com/neuro.rss",
        "https://www.nature.com/nmat.rss",
        "https://www.nature.com/nphys.rss",
        "https://www.nature.com/nchem.rss",
        "https://www.nature.com/ncomms.rss",
        "https://www.nature.com/natcomputsci.rss",
    ],
    "cell_press": [
        "https://www.cell.com/cell/rss",
        "https://www.cell.com/cancer-cell/rss",
        "https://www.cell.com/neuron/rss",
        "https://www.cell.com/chem/rss",
        "https://www.cell.com/cell-metabolism/rss",
    ],
    "science_aaas": [
        "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science",
    ],
    "pnas": [
        "https://www.pnas.org/action/showFeed?type=etoc&feed=rss&jc=PNAS",
    ],
    "peter_attia": ["https://peterattiadrive.libsyn.com/rss"],
    "science_daily": [
        "https://www.sciencedaily.com/rss/top/health.xml",
        "https://www.sciencedaily.com/rss/top/science.xml",
        "https://www.sciencedaily.com/rss/matter_energy/materials_science.xml",
        "https://www.sciencedaily.com/rss/matter_energy/physics.xml",
        "https://www.sciencedaily.com/rss/matter_energy/chemistry.xml",
        "https://www.sciencedaily.com/rss/matter_energy/engineering.xml",
        "https://www.sciencedaily.com/rss/matter_energy/nanotechnology.xml",
        "https://www.sciencedaily.com/rss/matter_energy/quantum_physics.xml",
        "https://www.sciencedaily.com/rss/matter_energy/biochemistry.xml",
        "https://www.sciencedaily.com/rss/computers_math/computer_science.xml",
        "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        "https://www.sciencedaily.com/rss/earth_climate/earth_science.xml",
        "https://www.sciencedaily.com/rss/space_time/astronomy.xml",
        "https://www.sciencedaily.com/rss/space_time/astrophysics.xml",
    ],
    "medrxiv": _extract_medrxiv_urls(_MEDRXIV_HTML),
    "biorxiv": [
        f"http://connect.biorxiv.org/biorxiv_xml.php?subject={subj}"
        for subj in [
            "neuroscience", "genetics", "genomics", "immunology",
            "cancer_biology", "biochemistry", "biophysics", "cell_biology",
            "microbiology", "molecular_biology", "pharmacology_and_toxicology",
            "synthetic_biology", "systems_biology", "ecology",
            "evolutionary_biology", "developmental_biology", "bioinformatics",
            "bioengineering", "physiology", "plant_biology", "pathology",
            "clinical_trials", "paleontology", "zoology",
            "animal_behavior_and_cognition",
            "scientific_communication_and_education",
        ]
    ],
    "arxiv": [
        f"https://rss.arxiv.org/rss/{cat}"
        for cat in [
            "cond-mat.mtrl-sci", "cond-mat.mes-hall", "cond-mat.soft",
            "cond-mat.supr-con", "physics.chem-ph", "physics.app-ph",
            "physics.optics", "cs.AI", "cs.LG", "cs.CV",
            "q-bio.BM", "q-bio.GN",
        ]
    ],
}

# ---------------------------------------------------------------------------
# Specialty lookup maps (subject name -> URL)
# ---------------------------------------------------------------------------

_MEDRXIV_SPECIALTY_URLS: dict[str, str] = {}
for _url in ALL_AVAILABLE_FEEDS["medrxiv"]:
    _match = re.search(r"subject=([^&]+)", _url)
    if _match:
        _MEDRXIV_SPECIALTY_URLS[_match.group(1)] = _url

_BIORXIV_SPECIALTY_URLS: dict[str, str] = {}
for _url in ALL_AVAILABLE_FEEDS["biorxiv"]:
    _match = re.search(r"subject=([^&]+)", _url)
    if _match:
        _BIORXIV_SPECIALTY_URLS[_match.group(1)] = _url

_ARXIV_CATEGORY_URLS: dict[str, str] = {}
for _url in ALL_AVAILABLE_FEEDS["arxiv"]:
    _cat = _url.rsplit("/", 1)[-1]
    _ARXIV_CATEGORY_URLS[_cat] = _url

_SPECIALTY_CATEGORIES = {"medrxiv", "biorxiv", "arxiv"}

# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S %Z",
    "%a, %d %b %Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def parse_date(date_str: str) -> datetime:
    if not date_str:
        return datetime.min
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    logger.warning("Could not parse date '%s'", date_str)
    return datetime.min


def is_recent(date_str: str, days_back: int = 7) -> bool:
    dt = parse_date(date_str)
    if dt == datetime.min:
        return False
    return dt >= datetime.now() - timedelta(days=days_back)


# ---------------------------------------------------------------------------
# Feed resolution (source + category -> URL dict)
# ---------------------------------------------------------------------------

def resolve_feeds(
    source: str | None = None,
    category: str | None = None,
    user_prefs: dict | None = None,
) -> dict[str, list[str]]:
    """Build the feed URL dict for a given source/category/user-prefs combo.

    - source=None, category=None -> user's configured feeds (or all non-specialty)
    - source="arxiv", category=None -> all arxiv feeds in catalog
    - source="arxiv", category="cs.AI" -> just that one feed
    - source="biorxiv", category="neuroscience" -> just that one feed
    - source="medrxiv", category="Oncology" -> just that one feed
    - source="nature", category=None -> all nature journal feeds
    """
    # Single source + single subcategory
    if source and category:
        url = _resolve_single(source, category)
        if url:
            return {source: [url]}
        return {}

    # Single source, all subcategories
    if source:
        if source in ALL_AVAILABLE_FEEDS:
            return {source: list(ALL_AVAILABLE_FEEDS[source])}
        return {}

    # No source specified -> use user prefs or defaults
    if user_prefs:
        return _resolve_from_prefs(user_prefs)

    # Default: all non-specialty sources
    return {
        k: list(v)
        for k, v in ALL_AVAILABLE_FEEDS.items()
        if k not in _SPECIALTY_CATEGORIES
    }


def _resolve_single(source: str, category: str) -> str | None:
    if source == "medrxiv":
        return _MEDRXIV_SPECIALTY_URLS.get(category)
    if source == "biorxiv":
        return _BIORXIV_SPECIALTY_URLS.get(category)
    if source == "arxiv":
        return _ARXIV_CATEGORY_URLS.get(category)
    # For non-specialty sources there's no subcategory concept
    return None


def _resolve_from_prefs(prefs: dict) -> dict[str, list[str]]:
    """Build feeds dict from user preference fields."""
    feeds: dict[str, list[str]] = {}

    rss_categories = prefs.get("rss_categories") or []
    cats = rss_categories or [
        k for k in ALL_AVAILABLE_FEEDS if k not in _SPECIALTY_CATEGORIES
    ]
    for cat in cats:
        if cat in ALL_AVAILABLE_FEEDS and cat not in _SPECIALTY_CATEGORIES:
            feeds[cat] = list(ALL_AVAILABLE_FEEDS[cat])

    medrxiv_specs = prefs.get("medrxiv_specialties") or []
    if medrxiv_specs:
        urls = [_MEDRXIV_SPECIALTY_URLS[s] for s in medrxiv_specs if s in _MEDRXIV_SPECIALTY_URLS]
        if urls:
            feeds["medrxiv"] = urls

    biorxiv_specs = prefs.get("biorxiv_specialties") or []
    if biorxiv_specs:
        urls = [_BIORXIV_SPECIALTY_URLS[s] for s in biorxiv_specs if s in _BIORXIV_SPECIALTY_URLS]
        if urls:
            feeds["biorxiv"] = urls

    arxiv_cats = prefs.get("arxiv_categories") or []
    if arxiv_cats:
        urls = [_ARXIV_CATEGORY_URLS[c] for c in arxiv_cats if c in _ARXIV_CATEGORY_URLS]
        if urls:
            feeds["arxiv"] = urls

    return feeds


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------

async def fetch_single_feed(
    category: str, url: str, days_back: int = 7,
) -> dict[str, Any]:
    """Fetch one RSS feed URL. Uses asyncio.to_thread for blocking feedparser."""
    logger.info("Fetching %s: %s", category, url)
    try:
        feed = await asyncio.to_thread(feedparser.parse, url)

        if hasattr(feed, "status") and feed.status != 200:
            logger.warning("HTTP %s for %s", feed.status, url)
            return {"category": category, "url": url, "feed_title": "", "entries": []}

        entries: list[dict[str, Any]] = []
        for entry in feed.entries:
            published = (
                entry.get("published", "")
                or entry.get("pubDate", "")
                or entry.get("updated", "")
            )
            if not published or not is_recent(published, days_back):
                continue

            authors_raw = entry.get("authors", [])
            if not authors_raw:
                author_str = entry.get("author", "") or entry.get("dc_creator", "")
                if author_str:
                    authors_raw = [{"name": n.strip()} for n in author_str.split(",") if n.strip()]

            item: dict[str, Any] = {
                "title": entry.get("title", "No title"),
                "link": entry.get("link", ""),
                "published": published,
                "summary": entry.get("summary", "") or entry.get("description", ""),
                "authors": authors_raw,
                "tags": [tag.term for tag in entry.get("tags", [])],
                "source_category": category,
            }

            for field in ("doi", "arxiv_id", "pmid"):
                if field in entry:
                    item[field] = entry[field]

            # Try to extract arXiv ID from link if category is arxiv
            if category == "arxiv" and "arxiv_id" not in item and item["link"]:
                aid_match = re.search(r"(\d{4}\.\d{4,5})", item["link"])
                if aid_match:
                    item["arxiv_id"] = aid_match.group(1)

            entries.append(item)

        logger.info(
            "%s: %d recent entries out of %d total",
            category, len(entries), len(feed.entries),
        )
        return {
            "category": category,
            "url": url,
            "feed_title": feed.feed.get("title", "Unknown"),
            "entries": entries,
        }
    except Exception as e:
        logger.error("Error fetching %s %s: %s", category, url, e)
        return {"category": category, "url": url, "feed_title": "", "entries": []}


async def fetch_feeds(
    feeds: dict[str, list[str]],
    days_back: int = 7,
    max_concurrent: int = 15,
) -> list[dict[str, Any]]:
    """Fetch multiple feeds in parallel with semaphore."""
    sem = asyncio.Semaphore(max_concurrent)
    pairs = [(cat, url) for cat, urls in feeds.items() for url in urls]

    async def _guarded(cat: str, url: str) -> dict[str, Any]:
        async with sem:
            return await fetch_single_feed(cat, url, days_back)

    results = await asyncio.gather(*[_guarded(c, u) for c, u in pairs])
    return list(results)
