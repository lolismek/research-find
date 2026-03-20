"""Minimal web chat interface — POC only, will be replaced by iMessage."""

from __future__ import annotations

import json
import asyncio

import anthropic
from aiohttp import web

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
</style>
</head>
<body>
<div id="messages"></div>
<div id="input-bar">
  <input id="msg" type="text" placeholder="Ask about papers..." autofocus />
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const msgs = document.getElementById('messages');
const inp = document.getElementById('msg');
const btn = document.getElementById('send');
let ws;

function connect() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'text') { addMsg(data.content, 'assistant'); btn.disabled = false; }
    else if (data.type === 'done') btn.disabled = false;
  };
  ws.onclose = () => setTimeout(connect, 1000);
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
  ws.send(text);
  inp.value = '';
  btn.disabled = true;
}

inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
connect();
</script>
</body>
</html>"""


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client = anthropic.AsyncAnthropic()
    messages = []
    monitor = get_monitor()

    print("[web] Client connected")

    async for raw in ws:
        if raw.type != web.WSMsgType.TEXT:
            break

        user_input = raw.data.strip()
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

                # Execute tools
                tool_results = []
                for tool_use in tool_uses:
                    print(f"[web]   calling {tool_use.name}({json.dumps(tool_use.input, default=str)[:80]})")
                    result_str = await dispatch_tool(tool_use.name, tool_use.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result_str,
                    })

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
