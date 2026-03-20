import feedparser
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def extract_medrxiv_urls_from_html(html_content: str) -> List[str]:
    """
    Extract all medRxiv RSS URLs from the HTML select options.
    """
    # Use regex to find all value attributes from option tags
    pattern = r'value="(http://connect\.medrxiv\.org/medrxiv_xml\.php\?subject=[^"]*)"'
    urls = re.findall(pattern, html_content)

    # Remove duplicates and filter out empty values
    unique_urls = list(set(url for url in urls if url and url != ""))

    return unique_urls


_MEDRXIV_HTML = """<select name="selectedPage" onchange="changePage(this.form.selectedPage)"><option value="">Select a subject category &nbsp;</option><option value="http://connect.medrxiv.org/medrxiv_xml.php?subject=all">All</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Addiction_Medicine">Addiction Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Allergy_and_Immunology">Allergy and Immunology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Anesthesia">Anesthesia</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Cardiovascular_Medicine">Cardiovascular Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Dentistry_and_Oral_Medicine">Dentistry and Oral Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Dermatology">Dermatology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Emergency_Medicine">Emergency Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=endocrinology">Endocrinology (including Diabetes Mellitus and Metabolic Disease)</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Epidemiology">Epidemiology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Forensic_Medicine">Forensic Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Gastroenterology">Gastroenterology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Genetic_and_Genomic_Medicine">Genetic and Genomic Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Geriatric_Medicine">Geriatric Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Health_Economics">Health Economics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Health_Informatics">Health Informatics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Health_Policy">Health Policy</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Health_Systems_and_Quality_Improvement">Health Systems and Quality Improvement</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Hematology">Hematology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=hivaids">HIV/AIDS</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=infectious_diseases">Infectious Diseases (except HIV/AIDS)</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Intensive_Care_and_Critical_Care_Medicine">Intensive Care and Critical Care Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Medical_Education">Medical Education</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Medical_Ethics">Medical Ethics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Nephrology">Nephrology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Neurology">Neurology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Nursing">Nursing</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Nutrition">Nutrition</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Obstetrics_and_Gynecology">Obstetrics and Gynecology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Occupational_and_Environmental_Health">Occupational and Environmental Health</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Oncology">Oncology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Ophthalmology">Ophthalmology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Orthopedics">Orthopedics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Otolaryngology">Otolaryngology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Pain_Medicine">Pain Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Palliative_Medicine">Palliative Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Pathology">Pathology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Pediatrics">Pediatrics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Pharmacology_and_Therapeutics">Pharmacology and Therapeutics</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Primary_Care_Research">Primary Care Research</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Psychiatry_and_Clinical_Psychology">Psychiatry and Clinical Psychology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Public_and_Global_Health">Public and Global Health</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Radiology_and_Imaging">Radiology and Imaging</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Rehabilitation_Medicine_and_Physical_Therapy">Rehabilitation Medicine and Physical Therapy</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Respiratory_Medicine">Respiratory Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Rheumatology">Rheumatology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Sexual_and_Reproductive_Health">Sexual and Reproductive Health</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Sports_Medicine">Sports Medicine</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Surgery">Surgery</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Toxicology">Toxicology</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Transplantation">Transplantation</option><option style="text-transform:capitalize;" value="http://connect.medrxiv.org/medrxiv_xml.php?subject=Urology">Urology</option> </select>"""

# Full catalog of all available feeds, keyed by category
ALL_AVAILABLE_FEEDS: Dict[str, List[str]] = {
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
    "medrxiv": extract_medrxiv_urls_from_html(_MEDRXIV_HTML),
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

# Map medRxiv specialty names → their RSS URLs for per-user filtering
_MEDRXIV_SPECIALTY_URLS: Dict[str, str] = {}
for _url in ALL_AVAILABLE_FEEDS["medrxiv"]:
    _match = re.search(r"subject=([^&]+)", _url)
    if _match:
        _MEDRXIV_SPECIALTY_URLS[_match.group(1)] = _url

# Map bioRxiv subject names → their RSS URLs for per-user filtering
_BIORXIV_SPECIALTY_URLS: Dict[str, str] = {}
for _url in ALL_AVAILABLE_FEEDS["biorxiv"]:
    _match = re.search(r"subject=([^&]+)", _url)
    if _match:
        _BIORXIV_SPECIALTY_URLS[_match.group(1)] = _url

# Map arXiv category codes → their RSS URLs for per-user filtering
_ARXIV_CATEGORY_URLS: Dict[str, str] = {}
for _url in ALL_AVAILABLE_FEEDS["arxiv"]:
    _cat = _url.rsplit("/", 1)[-1]
    _ARXIV_CATEGORY_URLS[_cat] = _url


def get_all_feed_urls() -> Dict[str, List[str]]:
    """
    Get all RSS feed URLs organized by category.
    Unchanged — used by pipeline.py for backward compat.
    """
    return dict(ALL_AVAILABLE_FEEDS)


_SPECIALTY_CATEGORIES = {"medrxiv", "biorxiv", "arxiv"}


def get_feed_urls_for_user(
    rss_categories: List[str] | None = None,
    medrxiv_specialties: List[str] | None = None,
    biorxiv_specialties: List[str] | None = None,
    arxiv_categories: List[str] | None = None,
    custom_urls: List[str] | None = None,
) -> Dict[str, List[str]]:
    """Build a filtered feed dict based on a user's preferences.

    Args:
        rss_categories: Which top-level categories to include (e.g. ["nature", "peter_attia"]).
                       If None or empty, includes all categories except specialty ones
                       (medrxiv, biorxiv, arxiv) which require explicit selection.
        medrxiv_specialties: Which medRxiv subjects (e.g. ["Oncology", "Cardiovascular_Medicine"]).
        biorxiv_specialties: Which bioRxiv subjects (e.g. ["neuroscience", "genetics"]).
        arxiv_categories: Which arXiv categories (e.g. ["cs.AI", "cond-mat.mtrl-sci"]).
        custom_urls: Extra feed URLs the user provided.

    Returns:
        Dict of category → list of URLs, suitable for passing to aggregate_all_feeds.
    """
    feeds: Dict[str, List[str]] = {}

    # RSS categories (exclude specialty categories that need explicit selection)
    cats = rss_categories or list(
        k for k in ALL_AVAILABLE_FEEDS if k not in _SPECIALTY_CATEGORIES
    )
    for cat in cats:
        if cat in ALL_AVAILABLE_FEEDS and cat not in _SPECIALTY_CATEGORIES:
            feeds[cat] = list(ALL_AVAILABLE_FEEDS[cat])

    # medRxiv specialties
    if medrxiv_specialties:
        medrxiv_urls = [
            _MEDRXIV_SPECIALTY_URLS[spec]
            for spec in medrxiv_specialties
            if spec in _MEDRXIV_SPECIALTY_URLS
        ]
        if medrxiv_urls:
            feeds["medrxiv"] = medrxiv_urls

    # bioRxiv specialties
    if biorxiv_specialties:
        biorxiv_urls = [
            _BIORXIV_SPECIALTY_URLS[spec]
            for spec in biorxiv_specialties
            if spec in _BIORXIV_SPECIALTY_URLS
        ]
        if biorxiv_urls:
            feeds["biorxiv"] = biorxiv_urls

    # arXiv categories
    if arxiv_categories:
        arxiv_urls = [
            _ARXIV_CATEGORY_URLS[cat]
            for cat in arxiv_categories
            if cat in _ARXIV_CATEGORY_URLS
        ]
        if arxiv_urls:
            feeds["arxiv"] = arxiv_urls

    # Custom URLs
    if custom_urls:
        feeds["custom"] = list(custom_urls)

    return feeds


def parse_date(date_str: str) -> datetime:
    """
    Parse various date formats commonly used in RSS feeds.
    Returns timezone-naive datetime for consistent comparison.
    """
    if not date_str:
        return datetime.min

    # Common RSS date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            # Convert to timezone-naive for consistent comparison
            if parsed_date.tzinfo:
                parsed_date = parsed_date.replace(tzinfo=None)
            return parsed_date
        except ValueError:
            continue

    # If all parsing fails, return a very old date
    print(f"Warning: Could not parse date '{date_str}'")
    return datetime.min


def is_recent(date_str: str, days_back: int = 7) -> bool:
    """
    Check if a date string represents a date within the last N days.
    """
    try:
        entry_date = parse_date(date_str)
        if entry_date == datetime.min:
            return False

        # Remove timezone info for comparison
        if entry_date.tzinfo:
            entry_date = entry_date.replace(tzinfo=None)

        cutoff_date = datetime.now() - timedelta(days=days_back)
        return entry_date >= cutoff_date
    except Exception as e:
        print(f"Error checking date recency: {e}")
        return False


def fetch_single_feed(feed_info: tuple) -> Dict[str, Any]:
    """
    Fetch a single RSS feed and return processed entries.
    feed_info is a tuple of (category, url)
    """
    category, url = feed_info
    print(f"Fetching {category}: {url}")

    try:
        feed = feedparser.parse(url)

        # Check if the feed parsing was successful
        if hasattr(feed, "status") and feed.status != 200:
            print(f"Error fetching feed: {url}, Status code: {feed.status}")
            return {
                "category": category,
                "url": url,
                "entries": [],
                "error": f"HTTP {feed.status}",
            }

        if hasattr(feed, "bozo") and feed.bozo:
            print(f"Warning: Malformed feed detected for {url}")

        # Process entries
        processed_entries = []
        for entry in feed.entries:
            # Check if entry is from the last week
            published_date = (
                entry.get("published", "")
                or entry.get("pubDate", "")
                or entry.get("updated", "")
            )

            if not published_date or not is_recent(published_date):
                continue

            processed_entry = {
                "title": entry.get("title", "No title"),
                "link": entry.get("link", ""),
                "published": published_date,
                "summary": entry.get("summary", "") or entry.get("description", ""),
                "authors": entry.get("authors", []),
                "tags": [tag.term for tag in entry.get("tags", [])],
                "source_url": url,
                "source_category": category,
            }

            # Add any additional fields that might be present
            for field in ["doi", "arxiv_id", "pmid"]:
                if field in entry:
                    processed_entry[field] = entry[field]

            processed_entries.append(processed_entry)

        feed_info = {
            "category": category,
            "url": url,
            "feed_title": feed.feed.get("title", "Unknown"),
            "feed_description": feed.feed.get("description", ""),
            "entries": processed_entries,
            "total_entries_fetched": len(feed.entries),
            "recent_entries_count": len(processed_entries),
        }

        print(
            f"✓ {category}: Found {len(processed_entries)} recent entries out of {len(feed.entries)} total"
        )
        return feed_info

    except Exception as e:
        print(f"✗ Error processing {category} feed {url}: {e}")
        return {"category": category, "url": url, "entries": [], "error": str(e)}


def aggregate_all_feeds(max_workers: int = 10, days_back: int = 7) -> Dict[str, Any]:
    """
    Aggregate all RSS feeds and return recent entries from the last week.
    """
    print(f"Starting RSS aggregation for the last {days_back} days...")
    start_time = time.time()

    all_feeds = get_all_feed_urls()

    # Flatten all feed URLs with their categories
    feed_list = []
    for category, urls in all_feeds.items():
        for url in urls:
            feed_list.append((category, url))

    print(f"Total feeds to process: {len(feed_list)}")

    # Process feeds in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feed = {
            executor.submit(fetch_single_feed, feed_info): feed_info
            for feed_info in feed_list
        }

        for future in as_completed(future_to_feed):
            result = future.result()
            results.append(result)

    # Organize results
    aggregated_data = {
        "metadata": {
            "aggregation_timestamp": datetime.now().isoformat(),
            "days_back": days_back,
            "total_feeds_processed": len(feed_list),
            "processing_time_seconds": round(time.time() - start_time, 2),
        },
        "feeds": results,
        "summary": {
            "total_recent_entries": sum(
                len(feed.get("entries", [])) for feed in results
            ),
            "feeds_by_category": {},
        },
    }

    # Create summary by category
    for result in results:
        category = result["category"]
        if category not in aggregated_data["summary"]["feeds_by_category"]:
            aggregated_data["summary"]["feeds_by_category"][category] = {
                "feed_count": 0,
                "total_entries": 0,
                "successful_feeds": 0,
            }

        aggregated_data["summary"]["feeds_by_category"][category]["feed_count"] += 1
        aggregated_data["summary"]["feeds_by_category"][category]["total_entries"] += (
            len(result.get("entries", []))
        )

        if "error" not in result:
            aggregated_data["summary"]["feeds_by_category"][category][
                "successful_feeds"
            ] += 1

    print(f"\n📊 Aggregation Summary:")
    print(f"Total feeds processed: {len(feed_list)}")
    print(
        f"Total recent entries found: {aggregated_data['summary']['total_recent_entries']}"
    )
    print(
        f"Processing time: {aggregated_data['metadata']['processing_time_seconds']} seconds"
    )

    for category, stats in aggregated_data["summary"]["feeds_by_category"].items():
        print(
            f"{category}: {stats['total_entries']} entries from {stats['successful_feeds']}/{stats['feed_count']} feeds"
        )

    return aggregated_data


def save_aggregated_data(data: Dict[str, Any], filename: str = None) -> str:
    """
    Save aggregated data to a JSON file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rss_aggregation_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Data saved to: {filename}")
    return filename


def get_recent_entries_summary(
    data: Dict[str, Any], limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get a summary of the most recent entries across all feeds.
    """
    all_entries = []

    for feed in data["feeds"]:
        for entry in feed.get("entries", []):
            entry_with_source = entry.copy()
            entry_with_source["feed_title"] = feed.get("feed_title", "Unknown")
            all_entries.append(entry_with_source)

    # Sort by publication date (most recent first)
    sorted_entries = sorted(
        all_entries, key=lambda x: parse_date(x.get("published", "")), reverse=True
    )

    return sorted_entries[:limit]


# Legacy function for backwards compatibility
def subscribe_to_rss(feed_url):
    """
    Original function - now wraps the new functionality for single feeds.
    """
    result = fetch_single_feed(("legacy", feed_url))

    if "error" in result:
        print(f"Error fetching feed: {feed_url}, Error: {result['error']}")
        return

    print(f"**Feed Title:** {result.get('feed_title', 'Unknown')}")
    print(f"**Feed URL:** {feed_url}")
    print(f"**Feed Description:** {result.get('feed_description', 'No description')}")

    entries = result.get("entries", [])
    print(f"\n**Recent Entries (last 7 days): {len(entries)}**")

    for entry in entries[:10]:  # Limit to first 10 for display
        print(f"  **Title:** {entry['title']}")
        print(f"  **Link:** {entry['link']}")
        print(f"  **Published Date:** {entry['published']}")
        if entry["summary"]:
            print(f"  **Summary:** {entry['summary'][:200]}...")
        print("-" * 20)


# Main execution
if __name__ == "__main__":
    # Run the full aggregation
    aggregated_data = aggregate_all_feeds(days_back=7)

    # Save to JSON file
    filename = save_aggregated_data(aggregated_data)

    # Show recent entries summary
    print("\n🔥 Most Recent Entries:")
    recent_entries = get_recent_entries_summary(aggregated_data, limit=5)

    for i, entry in enumerate(recent_entries, 1):
        print(f"\n{i}. **{entry['title']}**")
        print(f"   Source: {entry['feed_title']} ({entry['source_category']})")
        print(f"   Published: {entry['published']}")
        print(f"   Link: {entry['link']}")
        if entry["summary"]:
            print(f"   Summary: {entry['summary'][:150]}...")
