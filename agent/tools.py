"""Tool definitions (JSON schemas) for the Claude agent."""

TOOLS = [
    {
        "name": "search_papers",
        "description": (
            "Search for academic papers across Semantic Scholar, PubMed, and Europe PMC. "
            "Returns ranked results with citation counts, open access status, and venue info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (topic, keywords, or research question)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_paper",
        "description": (
            "Add a paper to the database by URL, DOI, arXiv ID, or title. "
            "Resolves the identifier, fetches metadata from Semantic Scholar, "
            "and optionally processes the PDF through GROBID for full-text extraction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Paper URL (arXiv, DOI, Semantic Scholar), DOI string, or title",
                },
                "process_pdf": {
                    "type": "boolean",
                    "description": "Whether to process the PDF through GROBID (default: false)",
                    "default": False,
                },
                "source": {
                    "type": "string",
                    "enum": ["manual", "recommended", "rss"],
                    "description": "How the paper was added: 'manual' (user gave URL/DOI/title), 'recommended' (from search results), 'rss' (from RSS digest). Default: manual",
                    "default": "manual",
                },
            },
            "required": ["identifier"],
        },
    },
    {
        "name": "get_paper_details",
        "description": (
            "Get details of a paper stored in the database. "
            "Search by DOI, arXiv ID, or title fragment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper"},
                "arxiv_id": {"type": "string", "description": "arXiv ID of the paper"},
                "title": {"type": "string", "description": "Title or title fragment to search"},
            },
        },
    },
    {
        "name": "list_stored_papers",
        "description": "List papers currently stored in the database, most recently added first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of papers to list (default: 20)",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "fetch_arxiv_papers",
        "description": (
            "Fetch the latest arXiv papers on demand. "
            "Can fetch a specific category or all of arXiv if no category is given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "arXiv category to fetch (e.g., 'cs.AI', 'cs.CL', 'stat.ML'). Omit to fetch all categories.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top papers to return (default: 5)",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "set_notification_time",
        "description": (
            "Set the daily arXiv digest notification time. "
            "The system automatically sends a digest of new papers at this time every day."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hour": {
                    "type": "integer",
                    "description": "Hour of the day (0-23) for the daily notification",
                },
                "minute": {
                    "type": "integer",
                    "description": "Minute of the hour (0-59, default: 0)",
                    "default": 0,
                },
            },
            "required": ["hour"],
        },
    },
    {
        "name": "find_similar_papers",
        "description": (
            "Find papers similar to a given paper using vector similarity search (OpenAI text-embedding-3-small). "
            "Requires the paper to be stored in the database with an embedding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the reference paper"},
                "arxiv_id": {"type": "string", "description": "arXiv ID of the reference paper"},
                "title": {"type": "string", "description": "Title of the reference paper"},
                "limit": {
                    "type": "integer",
                    "description": "Number of similar papers to return (default: 10)",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "follow_concept",
        "description": (
            "Follow a specific research concept/topic to track interests. "
            "Use when the user expresses interest in a research area."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concept_name": {
                    "type": "string",
                    "description": "The research concept or topic to follow (e.g., 'transformer architectures', 'protein folding')",
                },
            },
            "required": ["concept_name"],
        },
    },
]
