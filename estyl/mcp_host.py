from __future__ import annotations
import asyncio, json, os, sys, re
from typing import Dict, Any, List, Optional, Tuple
import datetime
import atexit
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# --------------------------- Config ---------------------------------
MODEL = "gpt-4o-mini"

# Hard cap on turns kept in the "LLM window" (user+assistant skeleton turns).
MAX_WINDOW_TURNS = 4  # tiny and constant

# Secondary safety cap by characters to avoid rare overflows
MAX_WINDOW_CHARS = 6000

HISTORY_FILE = "chat_history.txt"
# --------------------------------------------------------------------

client = AsyncOpenAI()

def load_style_yaml(path: str = "estyl/style_guide.yaml") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.warning("style_guide.yaml not found; continuing without it.")
        return ""

STYLE_YAML = load_style_yaml()

SYSTEM_PROMPT = f"""You are Estyl, a fashion shopping assistant powered by tools.

You have access to the following tool : `estyl_retrieve`:
- When calling estyl_retrieve, you must always include a non-empty `categories` list.
- Infer categories from the style, vibe, or budget in the user query. 
- It has two modes: "single" (retrieve items from a single category) and "outfit" (retrieve items using more than one category).
- If mode is "single", you must always suggest 10 items.
- If mode is "outfit", you must always suggest 5 outfits, each with articles depending on user budget, occasion and preferences.
- Call the tool whenever the user asks for product suggestions, searching, filtering, budgeted looks, or outfits.
- If a query is vague or missing details, you may ask at most 1–2 short clarifying questions.
- After showing results, you can continue asking refinements (e.g. “Want something more premium?”) and make follow-up tool calls.
- Always prefer action → ask → retrieve → refine.

## Output Formatting
Always output results with the following properties:
- title (string)
- price (float)
- product_url (string)
- image_url (string)
- Do not include extra commentary, markdown, or descriptions. Just bullets of items.
"""

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "estyl_retrieve",
            "description": "Retrieve category-based fashion items (single mode) or compose outfits (outfit mode).",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["single", "outfit"]},
                    "text_query": {"type": "string"},
                    "search_with": {"type": "string", "enum": ["Text", "Image", "Text + Image"]},
                    "image_b64": {"type": ["string", "null"]},
                    "gender": {"type": "string", "enum": ["male","female","unisex"]},
                    "categories": {"type": "array", "items": {"type": "string"},"description": "List of product categories. If unknown, ask user what items they are looking for or just need complete outfit."},
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

# ---------------------- Utilities & Managers ------------------------

def content_text(msg) -> str:
    """
    OpenAI SDK may return message.content as a string or a list of content parts.
    Normalize to plain text.
    """
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, dict):
                t = part.get("text")
                if t:
                    out.append(t)
    return "\n".join(out) if out else ""

def _trim(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"... [truncated {len(s)-max_chars} chars]"

def _msg_len_chars(m: Dict[str, Any]) -> int:
    c = m.get("content") or ""
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(part.get("text","")) for part in c if isinstance(part, dict))
    return 0

@dataclass
class ToolRecord:
    result_id: str
    tool_call_id: str
    name: str
    args: Dict[str, Any]
    output: str
    ts: str
    label: str = ""

class ToolStore:
    """
    In-memory cache of tool outputs available for reuse, without polluting the LLM context.
    """
    def __init__(self) -> None:
        self._records: Dict[str, ToolRecord] = {}
        self._by_call: Dict[str, str] = {}
        self._counter: int = 0

    def store(self, tool_call_id: str, name: str, args: Dict[str, Any], output: str) -> ToolRecord:
        self._counter += 1
        rid = f"R{self._counter}"
        label = self._make_label(name, args, output)
        rec = ToolRecord(
            result_id=rid,
            tool_call_id=tool_call_id,
            name=name,
            args=args,
            output=output,
            ts=datetime.datetime.now().isoformat(timespec="seconds"),
            label=label,
        )
        self._records[rid] = rec
        self._by_call[tool_call_id] = rid
        return rec

    def _make_label(self, name: str, args: Dict[str, Any], output: str) -> str:
        # Short human label e.g. "single | sneakers | 10 items"
        mode = args.get("mode", "")
        text_query = (args.get("text_query") or "").strip()
        cats = args.get("categories") or []
        # Heuristic: count lines that look like bullets to guess item count
        lines = [ln for ln in output.splitlines() if ln.strip()]
        item_like = sum(1 for ln in lines if re.match(r"^[\-\*\d]+\s", ln) or ("product_url" in ln and "title" in ln))
        cat_part = ",".join(cats[:3])
        if len(cats) > 3:
            cat_part += f"+{len(cats)-3}"
        q_part = (text_query[:30] + "…") if len(text_query) > 30 else text_query
        return f"{mode or name} | {cat_part or q_part or 'request'} | ~{item_like} items"

    def list(self) -> List[ToolRecord]:
        # Newest first
        return sorted(self._records.values(), key=lambda r: r.ts, reverse=True)

    def get(self, result_id: str) -> Optional[ToolRecord]:
        return self._records.get(result_id)

    def get_by_call(self, tool_call_id: str) -> Optional[ToolRecord]:
        rid = self._by_call.get(tool_call_id)
        return self.get(rid) if rid else None

class ContextWindow:
    """
    Maintains a tiny, constant-size message window for the LLM, while logging
    everything in full_history for auditing/export.
    """
    def __init__(self, system_prompt: str):
        self.system_msg = {"role": "system", "content": system_prompt}
        self.window_history: List[Dict[str, Any]] = [self.system_msg.copy()]
        self.full_history: List[Dict[str, Any]] = [self.system_msg.copy()]

    def _cap_by_turns(self) -> None:
        sys_msgs = [m for m in self.window_history if m.get("role") == "system"]
        others = [m for m in self.window_history if m.get("role") != "system"]
        keep = others[-(MAX_WINDOW_TURNS * 2):]
        self.window_history = sys_msgs[:1] + keep

    def _cap_by_chars(self) -> None:
        # Enforce an overall char budget for safety.
        base = [m for m in self.window_history if m.get("role") == "system"]
        others = [m for m in self.window_history if m.get("role") != "system"]
        total = sum(_msg_len_chars(m) for m in base)
        kept: List[Dict[str, Any]] = []
        for m in reversed(others):
            if total + _msg_len_chars(m) <= MAX_WINDOW_CHARS:
                kept.append(m)
                total += _msg_len_chars(m)
            else:
                break
        self.window_history = base + list(reversed(kept))

    # ---------------- public API ----------------

    def add_user(self, text: str) -> None:
        msg = {"role": "user", "content": text}
        self.window_history.append(msg)
        self.full_history.append(msg)
        self._cap_by_turns()
        self._cap_by_chars()

    def add_assistant_window(self, text: str) -> None:
        """
        Add a SHORT assistant breadcrumb to the LLM window (not the full response).
        """
        msg = {"role": "assistant", "content": text}
        self.window_history.append(msg)
        self._cap_by_turns()
        self._cap_by_chars()

    def add_assistant_full(self, text: str) -> None:
        """
        Log the FULL assistant text to full history (for saving/audit) but do NOT
        add it to the LLM window (prevents token bloat).
        """
        msg = {"role": "assistant", "content": text}
        self.full_history.append(msg)

    def add_tool_msg_full(self, tool_call_id: str, name: str, args: Dict[str, Any], output: str) -> None:
        """
        Log the tool call and the tool output ONLY to full_history.
        """
        tool_call_msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}
            }]
        }
        tool_content_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": output,
        }
        self.full_history.append(tool_call_msg)
        self.full_history.append(tool_content_msg)

    def get_window(self) -> List[Dict[str, Any]]:
        # The messages we actually send to the LLM.
        return self.window_history

    def flush_to_file(self, path: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                for msg in self.full_history:
                    role = msg.get("role")
                    if role == "tool":
                        # Keep full tool outputs in logs
                        content = msg.get("content") or ""
                    else:
                        content = content_text(msg)
                    f.write(f"{role}: {content}\n")
            print(f"\nFull chat history saved to {path}")
        except Exception as e:
            print(f"Failed to save chat history: {e}")

# ---------------------------- Chat Loop -----------------------------

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

            ctx = ContextWindow(SYSTEM_PROMPT)
            tools = ToolStore()

            print("Estyl chat ready. Type your message. Ctrl+C or /exit to quit.")
            def save_history_to_file():
                ctx.flush_to_file(HISTORY_FILE)

            atexit.register(save_history_to_file)

            while True:
                try:
                    user = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Bye!")
                    save_history_to_file()
                    return
                if not user:
                    continue
                if user.lower() in {"/exit", "/quit"}:
                    print("👋 Bye!")
                    save_history_to_file()
                    return

                # ---------- local convenience commands (do not hit LLM) ----------
                if user.lower() == "/results":
                    records = tools.list()
                    if not records:
                        print("estyl> No cached results yet.\n")
                        continue
                    print("estyl> Cached results (newest first):")
                    for r in records:
                        print(f"  {r.result_id}  {r.ts}  {r.label}")
                    print()
                    continue

                m = re.match(r"^/show\s+(R\d+)$", user, flags=re.I)
                if m:
                    rid = m.group(1)
                    rec = tools.get(rid)
                    if not rec:
                        print(f"estyl> No result found for {rid}\n")
                        continue
                    print(f"estyl> [result {rid}]")
                    print(rec.output + "\n")
                    # Do not push this into the LLM context (console-only)
                    continue
                # ------------------------------------------------------------------

                # Append user message
                ctx.add_user(user)

                # Send minimal window to LLM
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=ctx.get_window(),
                    tools=OPENAI_TOOLS,
                )
                msg = resp.choices[0].message

                # Handle tool call (we only handle the first call, same as before)
                if getattr(msg, "tool_calls", None):
                    tc = msg.tool_calls[0]
                    if tc.function.name == "estyl_retrieve":
                        args = json.loads(tc.function.arguments or "{}")
                        tool_result = await session.call_tool("estyl_retrieve", args)
                        tool_output = tool_result.content[0].text if tool_result.content else "{}"

                        # Cache tool result WITHOUT polluting the LLM window
                        rec = tools.store(tc.id, "estyl_retrieve", args, tool_output)

                        # Log full details to full_history only
                        ctx.add_tool_msg_full(tc.id, "estyl_retrieve", args, tool_output)

                        # Show the real results to the user (console)
                        # print(f"estyl> {tool_output}\n")
                        # extracted = []
                        # for item in json.loads(tool_output).get("items", []):
                        #     props = item.get("properties", {})
                        #     extracted.append({
                        #         "uuid": item.get("uuid"),
                        #         "title": props.get("title"),
                        #         "brand": props.get("brand"),
                        #         "price": props.get("price"),
                        #         "image_url": props.get("gcs_image_path"),
                        #         "product_url": props.get("product_url"),
                        #     })
                        print(f"estyl > {tool_output}")

                        # Add a SHORT breadcrumb to the LLM window for future turns
                        # (This keeps the model aware something happened, without tokens.)
                        breadcrumb = f"(estyl_retrieve completed; stored as result_id={rec.result_id}, {rec.label})"
                        ctx.add_assistant_window(breadcrumb)

                        # Also log the full assistant "final answer" ONLY in full_history
                        ctx.add_assistant_full(tool_output)

                    else:
                        print("estyl> (unrecognized tool request)\n")
                        ctx.add_assistant_window("(unrecognized tool request)")
                        ctx.add_assistant_full("(unrecognized tool request)")

                else:
                    # Normal assistant response (keep window tiny; log full)
                    answer = content_text(msg) or ""
                    print(f"estyl> {answer}\n")
                    # Window gets a tiny breadcrumb only
                    brief = _trim(answer, 500)
                    ctx.add_assistant_window(brief)
                    # Full history gets the whole thing
                    ctx.add_assistant_full(answer)

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\n Bye!")
