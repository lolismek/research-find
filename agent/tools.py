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
        "name": "monitor_arxiv_topic",
        "description": (
            "Start monitoring an arXiv category for new papers. "
            "The system will poll the RSS feed and notify you of new papers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "arXiv category to monitor (e.g., 'cs.AI', 'cs.CL', 'stat.ML')",
                },
                "schedule": {
                    "type": "string",
                    "description": "When to deliver notifications: 'immediate' or a time like '9am' (default: immediate)",
                    "default": "immediate",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top papers to include in notifications (default: 5)",
                    "default": 5,
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "find_similar_papers",
        "description": (
            "Find papers similar to a given paper using vector similarity search (SPECTER embeddings). "
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
]
