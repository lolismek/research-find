"""Minimal web chat interface — POC only, will be replaced by iMessage."""

from __future__ import annotations

import json
import asyncio

import anthropic
from aiohttp import web

from agent.tools import TOOLS
from agent.handlers import dispatch_tool
from background.rss_monitor import get_monitor
from services.neo4j_store import store_user

SYSTEM_PROMPT = """\
You are a research discovery assistant. You help users find, organize, and explore \
academic papers. You have access to tools that let you:

- Search across Semantic Scholar, PubMed, and Europe PMC
- Add papers to a personal Neo4j database by URL, DOI, or title
- Process PDFs through GROBID for full-text extraction
- Fetch the latest arXiv papers on demand (by category or all)
- Receive a daily arXiv digest at a configurable time (default 9am)
- Set or change the daily notification time
- Find similar papers using OpenAI vector embeddings
- Follow specific research concepts/topics to track interests
- When adding papers, set source: "manual" (user gave URL/DOI/title), "recommended" (from search results), "rss" (from RSS digest)

The system automatically sends a daily arXiv digest. Users can also ask you to \
fetch papers right now using fetch_arxiv_papers, or change the daily notification \
time using set_notification_time.

When presenting search results, be concise but informative. Highlight key details \
like citation count, venue, and open access status. When the user asks to add a paper, \
confirm what was stored. Proactively suggest related searches or papers when relevant.

IMPORTANT: Only call add_paper when the user explicitly asks to add or save a paper. \
Do NOT add papers just because the user mentions or comments on them. Comments and \
opinions about papers should only trigger add_insight, not add_paper. Use \
get_paper_details first to check if a paper is already in the library before adding.

Format paper titles in **bold** and use structured lists for readability.

When a user comments on a paper (e.g., "I think the methodology is weak", "The use of \
attention here is really clever", "I disagree with their conclusions"), automatically \
call add_insight to record it. Do NOT ask the user for permission — just create the \
insight silently and continue the conversation naturally.

If a user comments on a paper that was mentioned earlier in the conversation (e.g., \
from search results or recommendations) and later decides to add it to the library, \
look back at the conversation context for any earlier comments about that paper and \
create insights for those comments at the same time.

When determining score_impact for an insight, follow these guidelines:
- Positive insights: +0.1 to +0.3 (minor praise), +0.3 to +0.5 (significant appreciation)
- Neutral observations: 0.0 (no score change)
- Negative insights: -0.1 to -0.2 (mild criticism), -0.2 to -0.4 (notable flaws), \
-0.4 to -0.7 (fundamental issues or disagreements)
The score on the ADDED edge represents how much the user values a paper (0.0 to 2.0, \
default 1.0). Adjust conservatively — a single negative comment shouldn't tank a paper.

When search results include a "user_interest_blurb" field, compare the search topic \
to the blurb. If the search query is related to the user's research interests described \
in the blurb, call search_papers again with the same query and personalize=true to get \
results re-ranked by personal relevance. If the search topic is outside the user's \
interests (the blurb seems unrelated), present the original results as-is."""

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>research-find</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  #messages { flex: 1; overflow-y: auto; padding: 1rem; }
  .msg { max-width: 720px; margin: 0 auto 0.75rem; padding: 0.6rem 0.9rem; border-radius: 8px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { background: #1a3a5c; margin-left: auto; text-align: right; max-width: 480px; }
  .msg.assistant { background: #1a1a1a; border: 1px solid #2a2a2a; }
.msg.assistant :is(strong, b) { color: #7cb3ff; }
  #input-bar { border-top: 1px solid #222; padding: 0.75rem; display: flex; gap: 0.5rem; max-width: 780px; width: 100%; margin: 0 auto; }
  #input-bar input { flex: 1; padding: 0.6rem 0.9rem; border-radius: 6px; border: 1px solid #333; background: #111; color: #e0e0e0; font-size: 0.95rem; outline: none; }
  #input-bar input:focus { border-color: #4a8af4; }
  #input-bar button { padding: 0.6rem 1.2rem; border-radius: 6px; border: none; background: #4a8af4; color: white; cursor: pointer; font-size: 0.95rem; }
  #input-bar button:disabled { opacity: 0.4; cursor: default; }
  #login { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 1rem; }
  #login input { padding: 0.6rem 0.9rem; border-radius: 6px; border: 1px solid #333; background: #111; color: #e0e0e0; font-size: 1rem; width: 260px; outline: none; text-align: center; }
  #login input:focus { border-color: #4a8af4; }
  #login button { padding: 0.6rem 1.8rem; border-radius: 6px; border: none; background: #4a8af4; color: white; cursor: pointer; font-size: 1rem; }
  #login h2 { color: #7cb3ff; font-weight: 500; }
  .hidden { display: none !important; }
</style>
</head>
<body>
<div id="login">
  <h2>research-find</h2>
  <input id="phone" type="tel" placeholder="Phone number" autofocus />
  <button onclick="startChat()">Start</button>
</div>
<div id="messages" class="hidden"></div>
<div id="input-bar" class="hidden">
  <input id="msg" type="text" placeholder="Ask about papers..." />
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const msgs = document.getElementById('messages');
const inp = document.getElementById('msg');
const btn = document.getElementById('send');
let ws;
let userPhone = null;

function startChat() {
  const phone = document.getElementById('phone').value.trim();
  if (!phone) return;
  userPhone = phone;
  document.getElementById('login').classList.add('hidden');
  msgs.classList.remove('hidden');
  document.getElementById('input-bar').classList.remove('hidden');
  inp.focus();
  connect();
}

document.getElementById('phone').addEventListener('keydown', (e) => { if (e.key === 'Enter') startChat(); });

function connect() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen = () => {
    ws.send(JSON.stringify({type: 'init', phone: userPhone}));
  };
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'text') { addMsg(data.content, 'assistant'); btn.disabled = false; }
    else if (data.type === 'done') btn.disabled = false;
  };
  ws.onclose = () => { if (userPhone) setTimeout(connect, 1000); };
}

function addMsg(text, cls) {
  // Minimal markdown: **bold**
  let html = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = html;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function send() {
  const text = inp.value.trim();
  if (!text || btn.disabled) return;
  addMsg(text, 'user');
  ws.send(JSON.stringify({type: 'message', content: text}));
  inp.value = '';
  btn.disabled = true;
}

inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
</script>
</body>
</html>"""


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client = anthropic.AsyncAnthropic()
    messages = []
    monitor = get_monitor()
    user_phone = None

    # Give the monitor a way to push messages directly to this chat
    async def send_to_chat(text: str):
        if not ws.closed:
            await ws.send_json({"type": "text", "content": text})

    monitor.set_send(send_to_chat)

    print("[web] Client connected")

    async for raw in ws:
        if raw.type != web.WSMsgType.TEXT:
            break

        raw_data = raw.data.strip()
        if not raw_data:
            continue

        # Parse structured JSON messages; fall back to plain text
        try:
            parsed = json.loads(raw_data)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if parsed and isinstance(parsed, dict):
            msg_type = parsed.get("type")
            if msg_type == "init":
                user_phone = parsed.get("phone")
                if user_phone:
                    try:
                        await store_user(user_phone)
                        monitor.set_user_phone(user_phone)
                        print(f"[web] User logged in: {user_phone}")
                    except Exception as e:
                        print(f"[web] Failed to store user: {e}")
                continue
            elif msg_type == "message":
                user_input = parsed.get("content", "").strip()
            else:
                user_input = raw_data
        else:
            user_input = raw_data

        if not user_input:
            continue

        print(f"[web] User: {user_input}")
        messages.append({"role": "user", "content": user_input})

        try:
            while True:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )

                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_uses = [b for b in assistant_content if b.type == "tool_use"]

                if not tool_uses:
                    text_parts = [b.text for b in assistant_content if hasattr(b, "text") and b.text]
                    full_text = "\n".join(text_parts) or "(no response)"
                    print(f"[web] Assistant: {full_text[:120]}...")
                    await ws.send_json({"type": "text", "content": full_text})
                    break

                # Execute tools in parallel
                for tool_use in tool_uses:
                    print(f"[web]   calling {tool_use.name}({json.dumps(tool_use.input, default=str)[:80]})")
                results = await asyncio.gather(*[
                    dispatch_tool(tu.name, tu.input, user_phone=user_phone)
                    for tu in tool_uses
                ])
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_str,
                    }
                    for tu, result_str in zip(tool_uses, results)
                ]

                messages.append({"role": "user", "content": tool_results})

        except Exception as e:
            print(f"[web] Error: {e}")
            await ws.send_json({"type": "text", "content": f"Error: {e}"})

    print("[web] Client disconnected")
    return ws


async def index_handler(request):
    return web.Response(text=HTML, content_type="text/html")


def create_app():
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    return app
