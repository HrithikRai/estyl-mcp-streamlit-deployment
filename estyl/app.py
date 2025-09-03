# estyl_streamlit_app.py
"""
Streamlit front-end for the MCP host (estyl).
Features implemented:
- Persistent MCP stdio client session that is started once and reused (doesn't restart per request)
- Background asyncio worker that handles LLM calls + tool calls and returns responses to Streamlit UI
- Upload up to 3 images per query (images embedded as base64 and sent to tools when needed)
- Fast UI: uses a request queue and background processing so the UI thread never restarts the MCP server
- Product collage view with image + title + price + product_url
- Minimal logging and history file support (HISTORY_FILE environment variable)

Run: streamlit run estyl_streamlit_app.py

Note: this is a single-file example focused on preserving the original runtime behaviour while
keeping the server persistent and the UI responsive. You may need to adapt import paths for mcp or
session APIs depending on your environment.
"""

from __future__ import annotations
import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv() 
import streamlit as st
from PIL import Image

# External dependencies from the original script
# These must be importable in the environment where this Streamlit app runs.
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------------------------------------------------------------------
# Configuration (same defaults as the original host)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HISTORY_FILE = os.getenv("HISTORY_FILE", "chat_history.txt")
MAX_EXCHANGES_IN_WINDOW = int(os.getenv("MAX_EXCHANGES_IN_WINDOW", "2"))
FIRST_PASS_MAX_TOKENS = int(os.getenv("FIRST_PASS_MAX_TOKENS", "64"))
SECOND_PASS_SUMMARIZE = os.getenv("SECOND_PASS_SUMMARIZE", "True").lower() in ("1","true","yes")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("estyl_streamlit")

SYSTEM_PROMPT = f"""You are Estyl, a fashion shopping assistant powered by tools.

You will sometimes be given up to 3 images alongside the user's text. Each image has:
- purpose: infer whether it's for "context" (style cues, fit, vibe) or "search" (image-vector retrieval).

Rules about images:
- Use images with purpose "context" only to enrich understanding (e.g., colors, style cues, fit).
- Use images with purpose "search" as query images for image-vector search (i.e., treat them as the image to match in the catalog).
- When calling the tool `estyl_retrieve`, if you want the tool to use an image, include an "images" array in the function arguments. Each image entry should include an "id" matching the provided ids. Optionally include "image_b64" directly if you want to pass the image inline. If you only include the id, the runtime will attach the corresponding base64 for you.

Tool usage rules (same as before):
- When calling estyl_retrieve, you must always include a non-empty `categories` list.
- Infer categories from user query (style, vibe, budget).
- Modes: "single" (10 items) or "outfit" (5 outfits).
- If missing details, ask at most 1–2 short clarifying questions.
- Always prefer action → ask → retrieve → refine.

## Output Formatting
Always output results with the following properties:
- title (string)
- price (float)
- product_url (string)
- image_url (string)
- Do not include extra commentary, markdown, or descriptions. Just bullets of items.
"""
# Tool schema (kept simple for the Streamlit frontend — the backend mcp server still enforces semantics)
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "estyl_retrieve",
            "description": "Retrieve category-based fashion items (single mode) or compose outfits (outfit mode). If images are provided, include them in 'images' array; each item can have id/path/purpose/image_b64.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["single", "outfit"]},
                    "text_query": {"type": "string"},
                    "search_with": {"type": "string", "enum": ["Text", "Image", "Text + Image"]},
                    "image_b64": {"type": ["string", "null"]},
                    "images": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "path": {"type": "string"},
                                "purpose": {"type": "string", "enum": ["context", "search"]},
                                "image_b64": {"type": ["string","null"]}
                            },
                        },
                        "description": "Optional array of images to use (id/path/purpose). If provided with id only, runtime will inject base64."
                    },
                    "gender": {"type": "string", "enum": ["male","female","unisex"]},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "brand_contains": {"type": ["string","null"]},
                    "budget": {"type": "number", "description": "User's budget, infer from chat.","default": 350},
                    "limit": {"type": "integer", "minimum": 10, "maximum": 50},
                    "topk_for_rerank": {"type": "integer", "minimum": 10, "maximum": 40},
                    "exclude_ids": {"type": ["array","null"], "items": {"type": "string"}},
                    "num_outfits": {"type": "integer", "minimum": 10, "maximum": 20, "description": "The number of outfits to compose, always 5."},
                    "articles": {"type": "integer", "minimum": 5, "maximum": 7},
                    "per_cat_candidates": {"type": "integer", "minimum": 5, "maximum": 10}
                },
                "required": ["mode", "text_query", "gender", "categories"],
            }
        }
    }
]

# ------------------------------------------------------------------
# Helpers: history logging, iso timestamps, base64 encoding

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_log(entry: Dict[str, Any]) -> None:
    try:
        if "images" in entry and isinstance(entry["images"], list):
            safe_images = []
            for im in entry["images"]:
                safe_images.append({k: v for k, v in im.items() if k != "image_b64"})
            entry = {**entry, "images": safe_images}
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write history: {e}")


def encode_image_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


# ------------------------------------------------------------------
# Background persistent MCP connector and worker

@dataclass
class ChatRequest:
    user_text: str
    images_meta: List[Dict[str, Any]]
    future: "asyncio.Future[str]"


class MCPWorker:
    """Runs an asyncio loop in a background thread, keeps the MCP session open and processes requests
    pushed to a request_queue. Each request returns an assistant response string via a Future.
    """

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.request_queue: Optional[asyncio.Queue] = None
        self._stop_event = threading.Event()
        # Use a small in-memory deque for last exchanges (same semantics as original host)
        self.last_exchanges: deque = deque(maxlen=MAX_EXCHANGES_IN_WINDOW)
        self.client: Optional[AsyncOpenAI] = None
        self.session: Optional[ClientSession] = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        # wait until ready
        while not (self.loop and self.request_queue):
            time.sleep(0.05)

    def _run_loop(self):
        # each worker has its own event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.request_queue = asyncio.Queue()
        self.loop.run_until_complete(self._main())

    async def _main(self):
        # initialize OpenAI client and mcp stdio session once and reuse
        self.client = AsyncOpenAI()

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "estyl.mcp_server"],
            env=os.environ.copy(),
        )

        # Keep the stdio_client open for the life of the worker
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    await session.list_tools()
                    logger.info("MCP session established and ready")
                    await self._process_requests()
        except Exception as e:
            logger.exception("MCPWorker failed to start MCP server: %s", e)

    async def _process_requests(self):
        while True:
            req: ChatRequest = await self.request_queue.get()
            try:
                answer = await self._handle_turn(req.user_text, req.images_meta)
                if not req.future.done():
                    req.future.set_result(answer)
            except Exception as e:
                logger.exception("Error processing request: %s", e)
                if not req.future.done():
                    req.future.set_result(f"(error: {e})")

    async def _handle_turn(self, user: str, images_meta: List[Dict[str, Any]]) -> str:
        # Log user turn
        append_log({"ts": now_iso(), "role": "user", "content": user, "images": images_meta})

        # Build messages window (system + last exchanges + current user + images metadata)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in list(self.last_exchanges):
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user})
        if images_meta:
            visible_meta = []
            for im in images_meta:
                visible_meta.append({"id": im["id"], "purpose": im.get("purpose"), "has_b64": bool(im.get("image_b64"))})
            msgs.append({"role": "user", "name": "images", "content": json.dumps({"images": visible_meta}, ensure_ascii=False)})

        # First-pass LLM planning call (fast)
        resp = await self.client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            tools=OPENAI_TOOLS,
            temperature=0.2,
            max_tokens=FIRST_PASS_MAX_TOKENS,
        )
        msg = resp.choices[0].message

        if getattr(msg, "tool_calls", None) and msg.tool_calls:
            # prepare mapping id -> image
            images_by_id = {im["id"]: im for im in images_meta}

            async def run_one(tc):
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                call_id = tc.id
                # Inject image base64 when requested
                try:
                    if isinstance(args.get("images"), list):
                        for im in args["images"]:
                            if isinstance(im, dict):
                                if not im.get("image_b64"):
                                    if "id" in im and im["id"] in images_by_id:
                                        im["image_b64"] = images_by_id[im["id"]].get("image_b64", "")
                                    elif "path" in im:
                                        # attempt to read path if accessible
                                        try:
                                            with open(im["path"], "rb") as f:
                                                im["image_b64"] = encode_image_bytes_to_b64(f.read())
                                        except Exception:
                                            im["image_b64"] = ""
                        search_imgs = [x for x in args["images"] if x.get("purpose") == "search"]
                        pick = search_imgs[0] if search_imgs else (args["images"][0] if args["images"] else None)
                        if pick and pick.get("image_b64"):
                            args.setdefault("image_b64", pick["image_b64"])
                    else:
                        if args.get("image_id") and args["image_id"] in images_by_id:
                            args["image_b64"] = images_by_id[args["image_id"]].get("image_b64", "")
                        if args.get("path") and not args.get("image_b64"):
                            try:
                                with open(args["path"], "rb") as f:
                                    args["image_b64"] = encode_image_bytes_to_b64(f.read())
                            except Exception:
                                args["image_b64"] = ""
                except Exception as _e:
                    logger.warning("Failed to prepare image args: %s", _e)

                try:
                    tool_result = await self.session.call_tool(name, args)
                    out_text = tool_result.content[0].text if tool_result.content else ""
                except Exception as e:
                    out_text = f"(tool {name} failed: {e})"

                safe_args = args.copy()
                if "images" in safe_args:
                    for im in safe_args["images"]:
                        im.pop("image_b64", None)
                safe_args.pop("image_b64", None)
                append_log({
                    "ts": now_iso(),
                    "role": "tool",
                    "tool": name,
                    "arguments": safe_args,
                    "call_id": call_id,
                    "images_used": [ {"id": i.get("id"), "path": i.get("path"), "purpose": i.get("purpose")} for i in images_meta ]
                })
                return {"name": name, "args": args, "out": out_text, "call_id": call_id}

            # Execute tool calls concurrently for speed
            results = await asyncio.gather(*[run_one(tc) for tc in msg.tool_calls])

            if not SECOND_PASS_SUMMARIZE:
                combined = "\n".join(r["out"].strip() for r in results if r["out"])
                combined = combined or "(no results)"
                append_log({"ts": now_iso(), "role": "assistant", "content": combined})
                self.last_exchanges.append((user, combined[:400]))
                return combined

            # Construct followup for the second pass
            followup_msgs = msgs.copy()
            followup_msgs.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                    }
                    for tc in (msg.tool_calls or [])
                ],
            })

            for r in results:
                followup_msgs.append({
                    "role": "tool",
                    "tool_call_id": r["call_id"],
                    "name": r["name"],
                    "content": r["out"],
                })

            follow_resp = await self.client.chat.completions.create(
                model=MODEL,
                messages=followup_msgs,
                temperature=0.2,
            )
            answer = (follow_resp.choices[0].message.content or "").strip() or "(no response)"
            append_log({"ts": now_iso(), "role": "assistant", "content": answer})
            self.last_exchanges.append((user, answer[:400]))
            return answer

        else:
            answer = (msg.content or "").strip() or "(no response from model)"
            append_log({"ts": now_iso(), "role": "assistant", "content": answer})
            self.last_exchanges.append((user, answer[:400]))
            return answer

    # Public method to submit a request from the main thread
    def submit(self, user_text: str, images_meta: List[Dict[str, Any]]) -> str:
        if not self.loop or not self.request_queue:
            raise RuntimeError("MCPWorker not started")
        fut = asyncio.run_coroutine_threadsafe(self._submit_async(user_text, images_meta), self.loop)
        # block until result ready (but in the UI we'd rather show spinner and not freeze)
        res = fut.result()
        return res

    async def _submit_async(self, user_text: str, images_meta: List[Dict[str, Any]]) -> str:
        future: asyncio.Future = self.loop.create_future()
        req = ChatRequest(user_text=user_text, images_meta=images_meta, future=future)
        await self.request_queue.put(req)
        # Wait for response (this will not block Streamlit main thread because call is run in background thread)
        return await future


# cached single worker across Streamlit runs
@st.cache_resource
def get_worker() -> MCPWorker:
    w = MCPWorker()
    w.start()
    return w


# ------------------------------------------------------------------
# Streamlit UI

def prepare_image_uploads(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
    images_meta: List[Dict[str, Any]] = []
    for idx, file in enumerate(uploaded_files[:3]):
        try:
            img = Image.open(file)
            # normalize and get bytes
            with open("/tmp/estyl_tmp_img_{}.jpg".format(idx), "wb") as out:
                img.save(out, format="JPEG")
            with open("/tmp/estyl_tmp_img_{}.jpg".format(idx), "rb") as f:
                b = f.read()
            b64 = encode_image_bytes_to_b64(b)
            images_meta.append({"id": f"img{idx+1}", "path": None, "purpose": None, "image_b64": b64, "preview": img})
        except Exception as e:
            logger.warning("Failed to process uploaded image: %s", e)
    return images_meta


def render_collage(products: List[Dict[str, Any]]):
    # products expected: list of dicts with title, price, product_url, image_url
    if not products:
        st.info("No products returned.")
        return
    cols = st.columns(3)
    for i, p in enumerate(products):
        col = cols[i % 3]
        with col:
            if p.get("image_url"):
                st.image(p["image_url"], use_column_width=True)
            st.markdown(f"**{p.get('title','Untitled')}**")
            if p.get('price') is not None:
                st.markdown(f"Price: ₹{p.get('price')}")
            if p.get('product_url'):
                st.markdown(f"[Open product]({p.get('product_url')})")


def main():
    st.set_page_config(page_title="Estyl — MCP Streamlit", layout="wide")
    st.title("Estyl — Fashion assistant (Streamlit)")
    st.write("Interact with the MCP-powered Estyl assistant. Upload up to 3 images to include in your query.")

    # left: chat & uploader, right: product / results pane
    left, right = st.columns([1, 2])
    with left:
        user_text = st.text_area("Your message", key="user_text", height=120)
        uploaded = st.file_uploader("Upload up to 3 images (optional)", accept_multiple_files=True, type=["png", "jpg", "jpeg"], key="uploader")
        images_meta = []
        if uploaded:
            images_meta = prepare_image_uploads(uploaded)
            cols = st.columns(len(images_meta))
            for i, im in enumerate(images_meta):
                with cols[i]:
                    st.image(im.get("preview"), caption=f"Image {i+1}")
        submit = st.button("Send")

    with right:
        st.header("Assistant response")
        response_box = st.empty()
        st.header("Products / Results")
        products_box = st.empty()

    worker = get_worker()

    if submit and user_text.strip():
        # show spinner while waiting for background processing — worker.submit blocks but runs in worker thread
        with st.spinner("Getting results from Estyl..."):
            try:
                # Submit to background worker
                result = worker.submit(user_text.strip(), images_meta)
            except Exception as e:
                result = f"(error submitting request: {e})"
        response_box.markdown(result)

        # Attempt to parse products out of the response if LLM returned a JSON-like bullets list
        products = []
        # Heuristic: try JSON decode first, else try to extract lines with 'title'/'price' style
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                products = parsed
        except Exception:
            # fallback: scan for lines like: title: ... price: ... image_url: ...
            cur = {}
            for line in result.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k in ("title", "name"):
                        cur.setdefault("title", v)
                    elif k == "price":
                        try:
                            cur.setdefault("price", float("".join([c for c in v if (c.isdigit() or c in ".,")]).replace(",","")))
                        except Exception:
                            cur.setdefault("price", v)
                    elif k in ("product_url", "url"):
                        cur.setdefault("product_url", v)
                    elif k in ("image_url", "image"):
                        cur.setdefault("image_url", v)
                # detect separators
                if len(cur) >= 2:
                    products.append(cur)
                    cur = {}
        # render collage
        render_collage(products)


if __name__ == "__main__":
    main()
