from __future__ import annotations
import asyncio, json, os, sys
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

MODEL = "gpt-4o-mini"
MAX_HISTORY_TURNS = 10          

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
- Call the tool whenever the user asks for product suggestions, searching, filtering, budgeted looks, or outfits.
- If a query is vague or missing details, you may ask at most 1–2 short clarifying questions.
- As soon as you have enough context to form a reasonable search (even with some fields missing), 
  immediately call the tool with the best arguments you can infer. 
- After showing results, you can continue asking refinements (e.g. “Want something more premium?”) and make follow-up tool calls.
- Always prefer action → ask → retrieve → refine.

- Free-chat if the user only asks for generic fashion advice with no need for catalog retrieval. Treat the STYLE GUIDE YAML below as binding rules during your entire conversation:
{STYLE_YAML}

## Inspiration Handling
- When the user asks for an inspiration (celebrity, city, aesthetic), always answer with 
  concrete outfit suggestions: 2–3 clothing items + optional accessory.
- Focus on clothing and fashion accessories (bags, shoes, jewelry).
- Do not suggest cosmetics, makeup, or beauty products.
- Phrase it as: “For a [X]-inspired look, consider …” and list pieces with a short reason.

## Off-Topic Handling
- If the user asks about something unrelated to fashion/clothing/outfits:
  • Politely refuse with a warm sentence.
  • Redirect gently back to fashion help.

## Other Behavior
- If user uses inappropriate or toxic language, respond with a light, style-focused redirect.
- If user rejects (“too expensive / not my style”), ask 1 decisive fix question.
- Mirror user’s language.

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
                    "budget_tier": {"type": "string", "enum": ["Budget","Mid","Premium","Luxury"]},
                    "budget": {"type": "number"},
                    "limit": {"type": "integer", "minimum": 10, "maximum": 50},
                    "topk_for_rerank": {"type": "integer", "minimum": 10, "maximum": 40},
                    "exclude_ids": {"type": ["array","null"], "items": {"type": "string"}},
                    "num_outfits": {"type": "integer", "minimum": 10, "maximum": 20, "description": "The number of outfits to compose, always 10."},
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
            history: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Estyl chat ready. Type your message. Ctrl+C or /exit to quit.")

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

                history.append({"role": "user", "content": user})
                history = cap_history(history)

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

                        final = await client.chat.completions.create(
                            model=MODEL,
                            messages=history,

                        )
                        answer = content_text(final.choices[0].message)
                        print(f"estyl> {answer}\n")
                        history.append({"role": "assistant", "content": answer})
                        history = cap_history(history)
                    else:
                        print("estyl> (unrecognized tool request)\n")
                        history.append({"role": "assistant", "content": "(unrecognized tool request)"})
                else:
                    answer = content_text(msg)
                    print(f"estyl> {answer}\n")
                    history.append({"role": "assistant", "content": answer})
                    history = cap_history(history)

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\n👋 Bye!")
