"""Entry point for the research discovery chat system."""

import sys

from dotenv import load_dotenv
from aiohttp import web


def main():
    load_dotenv()

    from services.neo4j_store import init_db, close as close_neo4j
    from web import create_app

    app = create_app()

    async def on_startup(app):
        try:
            await init_db()
            print("Connected to Neo4j.")
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j ({e}). Some features may be unavailable.")

    async def on_shutdown(app):
        await close_neo4j()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    port = 8000
    print(f"\n  Open http://localhost:{port} in your browser\n")
    web.run_app(app, host="localhost", port=port, print=None)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
