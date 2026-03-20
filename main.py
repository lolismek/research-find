"""Entry point for the research discovery chat system."""

import asyncio
import sys

from dotenv import load_dotenv


async def main():
    load_dotenv()

    # Import after dotenv so env vars are available
    from services.neo4j_store import init_db, close as close_neo4j
    from agent.chat import chat_loop

    try:
        await init_db()
        print("Connected to Neo4j.")
    except Exception as e:
        print(f"Warning: Could not connect to Neo4j ({e}). Some features may be unavailable.")

    try:
        await chat_loop()
    finally:
        await close_neo4j()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
