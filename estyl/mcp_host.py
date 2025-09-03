from __future__ import annotations
import asyncio, json, os, sys
from typing import Dict, Any, List
import datetime
import atexit
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

MODEL = "gpt-4o-mini"
MAX_HISTORY_TURNS = 5          
HISTORY_FILE = "chat_history.txt"

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
                "required": ["mode", "text_query", "gender"]
            }
        }
    }
]

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
            t = part.get("text") if isinstance(part, dict) else None
            if t:
                out.append(t)
        return "\n".join(out)
    return ""

def cap_history(history: List[Dict[str, Any]], max_turns: int = MAX_HISTORY_TURNS) -> List[Dict[str, Any]]:
    if not history:
        return history
    sys_msgs = [m for m in history if m.get("role") == "system"]
    others = [m for m in history if m.get("role") != "system"]
    keep = others[-(max_turns * 2):]
    return sys_msgs[:1] + keep

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
            full_history: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            history: List[Dict[str, Any]] = full_history.copy()  # capped for LLM
            print("Estyl chat ready. Type your message. Ctrl+C or /exit to quit.")
            def save_history_to_file():
                try:
                    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                        for msg in full_history:
                            role = msg.get("role")
                            content = content_text(msg)
                            f.write(f"{role}: {content}\n")
                    print(f"\n Full chat history saved to {HISTORY_FILE}")
                except Exception as e:
                    print(f"Failed to save chat history: {e}")
            
            atexit.register(save_history_to_file)
            while True:
                try:
                    user = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n Bye!")
                    save_history_to_file()
                    return
                if not user:
                    continue
                if user.lower() in {"/exit", "/quit"}:
                    print(" Bye!")
                    save_history_to_file()
                    return

                history.append({"role": "user", "content": user})
                history = cap_history(history)
                full_history.append({"role": "user", "content": user})
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=history,
                    tools=OPENAI_TOOLS,
                )
                msg = resp.choices[0].message

                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    if tc.function.name == "estyl_retrieve":
                        args = json.loads(tc.function.arguments or "{}")
                        result = await session.call_tool("estyl_retrieve", args)
                        tool_output = result.content[0].text if result.content else "{}"
                        history.append({
                            "role": "assistant",
                            "tool_calls": [{
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": "estyl_retrieve", "arguments": json.dumps(args)}
                            }]
                        })
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": "estyl_retrieve",
                            "content": tool_output,
                        })
                        history = cap_history(history)
                        full_history.append(history)

                        final = await client.chat.completions.create(
                            model=MODEL,
                            messages=history,

                        )
                        answer = content_text(final.choices[0].message)
                        print(f"estyl> {answer}\n")
                        history.append({"role": "assistant", "content": answer})
                        history = cap_history(history)
                        full_history.append(history)
                    else:
                        print("estyl> (unrecognized tool request)\n")
                        history.append({"role": "assistant", "content": "(unrecognized tool request)"})
                else:
                    answer = content_text(msg)
                    print(f"estyl> {answer}\n")
                    history.append({"role": "assistant", "content": answer})
                    history = cap_history(history)
                    full_history.append(history)

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\n Bye!")