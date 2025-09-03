from __future__ import annotations
import asyncio, json, os, sys
from typing import Dict, Any, List, Tuple
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import base64
import logging
import urllib.request

logging.basicConfig(level=logging.INFO)

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --------------------------- Config ---------------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HISTORY_FILE = os.getenv("HISTORY_FILE", "chat_history.txt")

# The model will only see these many past exchanges (user->assistant pairs).
MAX_EXCHANGES_IN_WINDOW = 2
# --------------------------------------------------------------------
IMAGE_PATHS = [
    "images/wedding.jpeg",
    "images/oxfords.jpeg"
]

client = AsyncOpenAI()

# ------------------------- System Prompt ----------------------------
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
last_exchanges: deque[Tuple[str, str]] = deque(maxlen=MAX_EXCHANGES_IN_WINDOW)

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
    Example:
        "/images ./shirt1.jpg, ./shoe1.png"
    Returns list of dicts with id, path, and base64.
    """
    images = []
    if not raw:
        return images

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for idx, path in enumerate(parts[:max_images]):
        if not os.path.exists(path):
            print(f"⚠️ Skipping missing image: {path}")
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
            print(f"❌ Error loading {path}: {e}")
            continue

    return images

# --------------------------------------------------------------------

# --- config toggles for latency ---
SECOND_PASS_SUMMARIZE = True   # set True only if you want LLM to rewrite tool output
FIRST_PASS_MAX_TOKENS = 64      # keep the planner cheap+snappy

async def run_chat():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "estyl.mcp_server"],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.list_tools()

            print("Estyl ready. Type your message. /exit to quit.")
            pending_images: List[Dict[str, Any]] = []
            while True:
                try:
                    user = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Bye!")
                    return

                if not user:
                    continue
                if user.lower() in {"/exit", "/quit"}:
                    print("👋 Bye!")
                    return

                # allow user to inject images dynamically
                if user.startswith("/images "):
                    raw = user[len("/images "):].strip()
                    pending_images = prepare_images_from_paths(raw, max_images=3)
                    print(f"📸 Loaded {len(pending_images)} image(s). They’ll be used for your next query only.")
                    continue

                images_meta = pending_images
                pending_images = []


                # Log user turn immediately (without image base64)
                append_log({"ts": now_iso(), "role": "user", "content": user, "images": images_meta})

                # Build minimal window (only last two exchanges) and include images metadata
                msgs = build_messages(user, None, images_meta)

                # Fast LLM call — no extra params, tool_choice auto
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=msgs,
                    tools=OPENAI_TOOLS,
                    temperature=0.2,
                    max_tokens=FIRST_PASS_MAX_TOKENS,
                )
                msg = resp.choices[0].message

                if getattr(msg, "tool_calls", None) and msg.tool_calls:
                    # prepare a mapping id -> image meta for runtime injection
                    images_by_id = {im["id"]: im for im in images_meta}

                    # Run all tool calls in parallel
                    async def run_one(tc):
                        name = tc.function.name
                        args = json.loads(tc.function.arguments or "{}")
                        call_id = tc.id  # <-- needed to link tool result back

                        # Ensure any image references in args are replaced with base64 where possible.
                        try:
                            # If an "images" array is present in the function args, inject base64 where missing.
                            if isinstance(args.get("images"), list):
                                for im in args["images"]:
                                    # im may be {"id": "img1"} or {"path": "..."} or already include image_b64
                                    if isinstance(im, dict):
                                        if not im.get("image_b64"):
                                            # try resolve by id or path from our local images_by_id
                                            if "id" in im and im["id"] in images_by_id:
                                                im["image_b64"] = images_by_id[im["id"]].get("image_b64", "")
                                            elif "path" in im:
                                                im["image_b64"] = encode_image_to_b64(im["path"])
                                # also, for convenience, set a top-level image_b64 for legacy tools if mode expects single image:
                                # pick first image marked as purpose == 'search', else first image
                                search_imgs = [x for x in args["images"] if x.get("purpose") == "search"]
                                pick = search_imgs[0] if search_imgs else (args["images"][0] if args["images"] else None)
                                if pick and pick.get("image_b64"):
                                    args.setdefault("image_b64", pick["image_b64"])
                            else:
                                # legacy single image arg: maybe user passed "image_id" or expects us to attach image_b64
                                if args.get("image_id") and args["image_id"] in images_by_id:
                                    args["image_b64"] = images_by_id[args["image_id"]].get("image_b64", "")
                                # or if function expects image_b64 but user provided image path
                                if args.get("path") and not args.get("image_b64"):
                                    args["image_b64"] = encode_image_to_b64(args["path"])
                        except Exception as _e:
                            logging.warning(f"Failed to prepare image args for tool call: {_e}")

                        try:
                            tool_result = await session.call_tool(name, args)
                            out_text = tool_result.content[0].text if tool_result.content else ""
                        except Exception as e:
                            out_text = f"(tool {name} failed: {e})"
                        # Log the full tool interaction but strip base64
                        safe_args = args.copy()
                        if "images" in safe_args:
                            for im in safe_args["images"]:
                                im.pop("image_b64", None)
                            # also if top-level image_b64 exists, remove it
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

                    # ======= A) Zero second pass: print tool output directly =======
                    if not SECOND_PASS_SUMMARIZE:
                        combined = "\n".join(r["out"].strip() for r in results if r["out"])
                        combined = combined or "(no results)"
                        print(f"estyl> {combined}\n")

                        append_log({"ts": now_iso(), "role": "assistant", "content": combined})
                        last_exchanges.append((user, combined[:400]))
                        continue  # done with this turn

                    # ======= B) Correct second pass with tool_call_id linking =======
                    followup_msgs = msgs.copy()

                    # include the assistant message that contained tool_calls
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

                    # append one tool message per result with tool_call_id
                    for r in results:
                        followup_msgs.append({
                            "role": "tool",
                            "tool_call_id": r["call_id"],  # <-- REQUIRED
                            "name": r["name"],
                            "content": r["out"],
                        })

                    follow_resp = await client.chat.completions.create(
                        model=MODEL,
                        messages=followup_msgs,
                        temperature=0.2,
                    )
                    answer = (follow_resp.choices[0].message.content or "").strip() or "(no response)"
                    print(f"estyl> {answer}\n")

                    append_log({"ts": now_iso(), "role": "assistant", "content": answer})
                    last_exchanges.append((user, answer[:400]))

                else:
                    # ✅ Plain text response from LLM (no tools used)
                    answer = (msg.content or "").strip() or "(no response from model)"
                    print(f"estyl> {answer}\n")
                    append_log({"ts": now_iso(), "role": "assistant", "content": answer})
                    last_exchanges.append((user, answer[:400]))

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\nBye!")
