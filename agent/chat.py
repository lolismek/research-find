"""Claude-powered chat loop with tool dispatch."""

from __future__ import annotations

import json

import anthropic

from agent.tools import TOOLS
from agent.handlers import dispatch_tool
from background.rss_monitor import get_monitor

SYSTEM_PROMPT = """\
You are a research discovery assistant. You help users find, organize, and explore \
academic papers. You have access to tools that let you:

- Search across Semantic Scholar, PubMed, and Europe PMC
- Add papers to a personal Neo4j database by URL, DOI, or title
- Process PDFs through GROBID for full-text extraction
- Monitor arXiv RSS feeds for new papers in specific categories
- Find similar papers using SPECTER vector embeddings

When presenting search results, be concise but informative. Highlight key details \
like citation count, venue, and open access status. When the user asks to add a paper, \
confirm what was stored. Proactively suggest related searches or papers when relevant.

Format paper titles in **bold** and use structured lists for readability."""


async def chat_loop():
    """Run the interactive chat loop."""
    client = anthropic.AsyncAnthropic()
    messages = []
    monitor = get_monitor()

    print("\n=== Research Discovery Chat ===")
    print("Ask me to find papers, add papers to your database, or monitor arXiv topics.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        # Check for pending RSS notifications
        notifications = monitor.get_pending_notifications()
        if notifications:
            for notif in notifications:
                print(f"\n[RSS Update] New papers in {notif['category']}:")
                for p in notif.get("papers", []):
                    print(f"  - {p.get('title', 'Untitled')}")
                print()

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Run the tool-use loop
        while True:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # Collect assistant content blocks
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check if there are tool uses
            tool_uses = [b for b in assistant_content if b.type == "tool_use"]

            if not tool_uses:
                # No more tool calls — print text response and break
                for block in assistant_content:
                    if hasattr(block, "text"):
                        print(f"\nAssistant: {block.text}\n")
                break

            # Print any intermediate text
            for block in assistant_content:
                if hasattr(block, "text") and block.text:
                    print(f"\nAssistant: {block.text}")

            # Execute tool calls
            tool_results = []
            for tool_use in tool_uses:
                print(f"  [calling {tool_use.name}...]")
                result_str = await dispatch_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})
