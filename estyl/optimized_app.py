from __future__ import annotations
import asyncio, json, os, sys, base64, logging, urllib.request, threading, queue, tempfile
from typing import Dict, Any, List, Tuple, Deque
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --------------------------- Config ---------------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HISTORY_FILE = os.getenv("HISTORY_FILE", "chat_history.txt")

# The model will only see these many past exchanges (user->assistant pairs).
MAX_EXCHANGES_IN_WINDOW = 2
# --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

client = AsyncOpenAI()

# ------------------------- System Prompt ----------------------------
SYSTEM_PROMPT = f"""You are Estyl, a fashion shopping assistant powered by tools.

You will sometimes be given up to 3 images alongside the user's text. Each image has:
- purpose: infer whether it's for "context" (style cues, fit, vibe) or "search" (image-vector retrieval).

Rules about images:
- Use images with purpose "context" only to enrich understanding (e.g., colors, style cues, fit).
- Use images with purpose "search" as query images for image-vector search (i.e., treat them as the image to match in the catalog).
- When calling the tool `estyl_retrieve`, if you want the tool to use an image, include an "images" array in the function arguments. Each image entry should include an "id" matching the provided ids. Optionally include "image_b64" directly if you want to pass the image inline. If you only include the id, the runtime will attach the corresponding base64 for you.

Tool usage rules:
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
# --------------------------------------------------------------------

# ------------------------- Tool Schemas -----------------------------
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
# --------------------------------------------------------------------

# ------------------------- Minimal History --------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def append_log(entry: Dict[str, Any]) -> None:
    # Avoid logging raw base64 for privacy/size reasons.
    try:
        # If images field exists, strip base64 before logging
        if "images" in entry and isinstance(entry["images"], list):
            safe_images = []
            for im in entry["images"]:
                safe_images.append({k: v for k, v in im.items() if k != "image_b64"})
            entry = {**entry, "images": safe_images}
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"Failed to write history: {e}")

# Each element is a (user_text, assistant_text) tuple.
last_exchanges: Deque[Tuple[str, str]] = deque(maxlen=MAX_EXCHANGES_IN_WINDOW)


def build_messages(user_msg: str, assistant_preview: str | None, images_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the messages window: [system] + last_exchanges + current user + images metadata message.
    images_meta is a list of dicts: {id, path, purpose, image_b64 (optional)}.
    """
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in list(last_exchanges):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})
    if images_meta:
        # Include a machine-readable images metadata message the LLM can inspect.
        # We include id/path/purpose and also include image size (bytes) but we avoid embedding full base64 in the visible metadata
        visible_meta = []
        for im in images_meta:
            visible_meta.append({
                "id": im["id"],
                "path": im.get("path"),
                "purpose": im.get("purpose"),
                "has_b64": bool(im.get("image_b64"))
            })
        msgs.append({
            "role": "user",
            "name": "images",
            "content": json.dumps({"images": visible_meta}, ensure_ascii=False)
        })
    return msgs

# -------------------------Image integration -------------------------
def encode_image_to_b64(path: str) -> str:
    """
    Accepts local path or http(s) URL. Returns base64 string.
    """
    try:
        if path.startswith("http://") or path.startswith("https://"):
            with urllib.request.urlopen(path, timeout=15) as resp:
                data = resp.read()
        else:
            with open(path, "rb") as f:
                data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logging.warning(f"Could not read image at {path}: {e}")
        return ""


def prepare_images_from_paths(raw: str, max_images: int = 3) -> List[Dict[str, Any]]:
    """
    Take a comma-separated string of image paths entered at runtime.
    Returns list of dicts with id, path, and base64.
    """
    images: List[Dict[str, Any]] = []
    if not raw:
        return images

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for idx, path in enumerate(parts[:max_images]):
        if not os.path.exists(path):
            st.warning(f"Skipping missing image: {path}")
            continue
        try:
            b64 = encode_image_to_b64(path)
            images.append({
                "id": f"img{idx+1}",
                "path": path,
                "purpose": None,   # let LLM infer purpose
                "image_b64": b64
            })
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
            continue

    return images
import json, re
def safe_json_loads(s: str):
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Fallback: try fixing common issues
        fixed = s.strip()

        # Replace single quotes → double
        fixed = fixed.replace("'", '"')

        # Remove trailing commas
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

        try:
            return json.loads(fixed)
        except Exception:
            # As last resort: return empty dict
            return {}
# --------------------------------------------------------------------

# --- config toggles for latency ---
SECOND_PASS_SUMMARIZE = True   # keep identical default
FIRST_PASS_MAX_TOKENS = 64      # keep the planner cheap+snappy

# ======================= Persistent MCP Worker =======================
class MCPWorker:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.started = threading.Event()
        self.session: ClientSession | None = None
        self.read = None
        self.write = None
        self.request_q: asyncio.Queue | None = None
        self._stop = False

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()
            self.started.wait(timeout=15)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._bootstrap())
        self.started.set()
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self._teardown())

    async def _bootstrap(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "estyl.mcp_server"],
            env=os.environ.copy(),
        )
        self.request_q = asyncio.Queue()
        # open persistent stdio client + session
        self._client_ctx = stdio_client(params)
        self.read, self.write = await self._client_ctx.__aenter__()
        self._session_ctx = ClientSession(self.read, self.write)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()
        await self.session.list_tools()
        # start worker consumer
        self.loop.create_task(self._consume())

    async def _teardown(self):
        try:
            if self.session is not None:
                await self._session_ctx.__aexit__(None, None, None)
            if hasattr(self, "_client_ctx"):
                await self._client_ctx.__aexit__(None, None, None)
        except Exception:
            pass

    async def _consume(self):
        while not self._stop:
            req = await self.request_q.get()
            if req is None:
                continue
            user, images_meta, fut = req["user"], req["images"], req["fut"]
            try:
                result = await self._handle_turn(user, images_meta)
                self.loop.call_soon_threadsafe(fut.set_result, result)
            except Exception as e:
                self.loop.call_soon_threadsafe(fut.set_exception, e)

    async def _handle_turn(self, user: str, images_meta: List[Dict[str, Any]]):
        # Log user turn immediately (without image base64)
        append_log({"ts": now_iso(), "role": "user", "content": user, "images": images_meta})

        # Build minimal window (only last two exchanges) and include images metadata
        msgs = build_messages(user, None, images_meta)

        resp = await client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            tools=OPENAI_TOOLS,
            temperature=0.2,
            max_tokens=FIRST_PASS_MAX_TOKENS,
        )
        msg = resp.choices[0].message

        # Tool calls path
        if getattr(msg, "tool_calls", None) and msg.tool_calls:
            images_by_id = {im["id"]: im for im in images_meta}

            async def run_one(tc):
                name = tc.function.name
                args = safe_json_loads(tc.function.arguments or "{}")
                call_id = tc.id
                # Ensure any image references in args are replaced with base64 where possible.
                try:
                    if isinstance(args.get("images"), list):
                        for im in args["images"]:
                            if isinstance(im, dict):
                                if not im.get("image_b64"):
                                    if "id" in im and im["id"] in images_by_id:
                                        im["image_b64"] = images_by_id[im["id"]].get("image_b64", "")
                                    elif "path" in im:
                                        im["image_b64"] = encode_image_to_b64(im["path"])
                        search_imgs = [x for x in args["images"] if x.get("purpose") == "search"]
                        pick = search_imgs[0] if search_imgs else (args["images"][0] if args["images"] else None)
                        if pick and pick.get("image_b64"):
                            args.setdefault("image_b64", pick["image_b64"])
                    else:
                        if args.get("image_id") and args["image_id"] in images_by_id:
                            args["image_b64"] = images_by_id[args["image_id"]].get("image_b64", "")
                        if args.get("path") and not args.get("image_b64"):
                            args["image_b64"] = encode_image_to_b64(args["path"])
                except Exception as _e:
                    logging.warning(f"Failed to prepare image args for tool call: {_e}")

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

            results = await asyncio.gather(*[run_one(tc) for tc in msg.tool_calls])

            if not SECOND_PASS_SUMMARIZE:
                combined = "\n".join(r["out"].strip() for r in results if r["out"]) or "(no results)"
                append_log({"ts": now_iso(), "role": "assistant", "content": combined})
                last_exchanges.append((user, combined[:400]))
                return combined

            followup_msgs = msgs.copy()
            followup_msgs.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    } for tc in (msg.tool_calls or [])
                ],
            })
            for r in results:
                followup_msgs.append({
                    "role": "tool",
                    "tool_call_id": r["call_id"],
                    "name": r["name"],
                    "content": r["out"],
                })

            follow_resp = await client.chat.completions.create(
                model=MODEL,
                messages=followup_msgs,
                temperature=0.2,
            )
            answer = (follow_resp.choices[0].message.content or "").strip() or "(no response)"
            append_log({"ts": now_iso(), "role": "assistant", "content": answer})
            last_exchanges.append((user, answer[:400]))
            return answer
        else:
            answer = (msg.content or "").strip() or "(no response from model)"
            append_log({"ts": now_iso(), "role": "assistant", "content": answer})
            last_exchanges.append((user, answer[:400]))
            return answer

    def ask(self, user: str, images_meta: List[Dict[str, Any]]):
        fut: asyncio.Future = self.loop.create_future()   # ✅ Fix here
        assert self.request_q is not None
        self.loop.call_soon_threadsafe(
            self.request_q.put_nowait,
            {"user": user, "images": images_meta, "fut": fut},
        )
        # Wait synchronously for result
        return asyncio.run_coroutine_threadsafe(self._await_future(fut), self.loop).result()

    async def _await_future(self, fut: asyncio.Future):
        return await fut

# ======================= Streamlit UI ================================
st.set_page_config(page_title="Estyl – Fashion Assistant", page_icon="🧥", layout="wide")

# Persistent worker in session
if "worker" not in st.session_state:
    st.session_state.worker = MCPWorker()
    st.session_state.worker.start()

if "chat" not in st.session_state:
    st.session_state.chat: List[Dict[str, Any]] = []

st.title("🧥 Estyl – Fashion Shopping Assistant")
st.caption("Persistent MCP server in the background. Clean UI • Small images • No restarts per turn.")

col1, col2 = st.columns([2,1])
with col2:
    st.subheader("Images (optional)")
    img_paths_raw = st.text_input("Local image paths (comma-separated)", placeholder="images/wedding.jpeg, images/oxfords.jpeg")
    uploaded = st.file_uploader("Or upload up to 3 images", accept_multiple_files=True, type=["png","jpg","jpeg","webp"])
    tmp_paths: List[str] = []
    if uploaded:
        for f in uploaded[:3]:
            # save to temp so that existing prepare_images_from_paths can read
            suffix = os.path.splitext(f.name)[1] or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(f.read())
            tmp.flush()
            tmp_paths.append(tmp.name)
    if tmp_paths:
        join_tmp = ", ".join(tmp_paths)
        if img_paths_raw:
            img_paths_raw = f"{img_paths_raw}, {join_tmp}"
        else:
            img_paths_raw = join_tmp
    st.caption("Tip: Pass product/context shots. We'll keep them small.")

with col1:
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and m.get("items"):
                # Render any parsed items as compact cards
                for it in m["items"]:
                    with st.container(border=True):
                        st.markdown(f"**{it.get('title','')}** – ₹{it.get('price','')}")
                        if it.get("image_url"):
                            st.image(it["image_url"], width=160)
                        if it.get("product_url"):
                            st.markdown(f"[View]({it['product_url']})")
            st.markdown(m["content"]) if m.get("content") else None

    prompt = st.chat_input("Describe what you're shopping for…")

# Handle submit
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    # Prepare images (existing helper)
    images_meta = prepare_images_from_paths(img_paths_raw, max_images=3) if img_paths_raw else []
    # Preview selected images in a tiny strip
    if images_meta:
        cols = st.columns(min(3, len(images_meta)))
        for i, im in enumerate(images_meta):
            with cols[i]:
                try:
                    if im.get("path") and os.path.exists(im["path"]):
                        st.image(im["path"], width=120)
                except Exception:
                    pass

    # Ask worker
    try:
        answer = st.session_state.worker.ask(prompt, images_meta)
    except Exception as e:
        answer = f"(runtime error: {e})"

    # Try to parse bullet lines into item dicts to show small images cleanly
    items: List[Dict[str, Any]] = []
    for line in answer.splitlines():
        line = line.strip("- • ")
        if not line:
            continue
        # crude extractor for key-value pairs
        if ("title:" in line and "image_url:" in line) or line.startswith("{"):
            try:
                if line.startswith("{"):
                    obj = safe_json_loads(line)
                else:
                    # parse "title: x; price: y; product_url: z; image_url: q"
                    parts = [p.strip() for p in line.split(";") if p.strip()]
                    kv = {}
                    for p in parts:
                        if ":" in p:
                            k,v = p.split(":",1)
                            kv[k.strip()] = v.strip()
                    obj = {
                        "title": kv.get("title",""),
                        "price": kv.get("price",""),
                        "product_url": kv.get("product_url",""),
                        "image_url": kv.get("image_url",""),
                    }
                items.append(obj)
            except Exception:
                pass

    # Store and render assistant turn
    st.session_state.chat.append({"role":"user","content":prompt})
    with st.chat_message("assistant"):
        if items:
            for it in items[:12]:
                with st.container(border=True):
                    st.markdown(f"**{it.get('title','')}** – {it.get('price','')}")
                    if it.get("image_url"):
                        st.image(it["image_url"], width=160)
                    if it.get("product_url"):
                        st.markdown(f"[View]({it['product_url']})")
        st.markdown(answer)
    st.session_state.chat.append({"role":"assistant","content":answer,"items":items})
