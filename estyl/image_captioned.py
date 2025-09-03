# type: ignore
from __future__ import annotations
import asyncio, json, os, sys
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()  
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import base64
import logging
logging.basicConfig(level=logging.INFO)

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

MODEL = "gpt-4o-mini"
MAX_HISTORY_TURNS = 10          

client = AsyncOpenAI()


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
                    "text_query": {"type": "string"},                                   #color 
                    "search_with": {"type": "string", "enum": ["Text", "Image", "Text + Image"]},
                    "image_b64": {"type": ["string", "null"]},
                    "gender": {"type": "string", "enum": ["male","female","unisex"]},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "brand_contains": {"type": ["string","null"]},
                    "budget_tier": {"type": "string", "enum": ["Budget","Mid","Premium","Luxury"]},
                    "budget": {"type": "number", "description": "User's budget, infer from chat."},
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
def load_style_yaml(filename: str = "style_guide.yaml") -> str:
    base_dir = os.path.dirname(__file__)  
    path = os.path.join(base_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

style_yaml = load_style_yaml()

SYSTEM_PROMPT = f"""You are Estyl, a helpful fashion assistant.

IMPORTANT RULE:
- If the user asks about ANYTHING unrelated to fashion, clothing, or outfits 
  (such as food, travel, restaurants, politics, sports, etc.), you MUST refuse. 
- When refusing, always reply politely: 
  "Sorry, I can’t answer that one. I’m here to help with style and outfits."

You have access to the following tool : `estyl_retrieve`:
- It has two modes: "single" (retrieve items from a single category) and "outfit" (retrieve items using more than one category).
- Call the tool whenever the user asks for product suggestions, searching, filtering, budgeted looks, or outfits.
- If a query is vague or missing details, you may ask at most 1–2 short clarifying questions.
- After showing results, you can continue asking refinements (e.g. “Want something more premium?”) and make follow-up tool calls.
- Always prefer action → ask → retrieve → refine.

Below is your complete behavior/style guide in YAML.  
You must treat the STYLE GUIDE YAML below as binding rules.
For every reply, enforce mechanics, tone, and lexicon replacements from it.

STYLE GUIDE:
{style_yaml}
- Keep tone inclusive; respect gender & category purity.  
- Do not suggest boring or generic outfits. Each recommendation should feel stylish, intentional, unique, and interesting.  
- Do not repeat the same clothing piece in different categories or across turns,  
  unless the repetition is necessary for a specific occasion (e.g., a white shirt appearing in multiple business outfits).  
- Variations are encouraged (e.g., "silk blouse in navy", "chiffon blouse in cream") instead of reusing the exact same item.  

## Inspiration Handling
- When the user asks for an inspiration (celebrity, city, aesthetic), always answer with  
  concrete outfit suggestions: 2–3 clothing items + optional accessory.  
- Focus on clothing and fashion accessories (bags, shoes, jewelry).  
- Do not suggest cosmetics, makeup, or beauty products.  
- Phrase it as: “For a [X]-inspired look, consider …” and list pieces with a short reason.  

## Style advice must include:
- Introduce the basics of **color theory** and how to apply it in fashion.  
- Give **practical examples** of how colors, fabrics, and cuts enhance an outfit.  
- Use **style archetypes** where relevant (e.g., posh elegance, edgy clubbing, casual coffee date, night out glamour).  

## Act as a stylist and critic:
- Critique poor choices if the user’s request is seasonally, stylistically, or contextually inappropriate,  
  and suggest better alternatives.  
- Explicitly exclude that item and propose a suitable replacement.  
- Example: If asked for linen in winter, recommend warmer fabrics instead.  

## Output Formatting
Always output results with the following properties:
- title (string)
- price (float)
- product_url (string)
- image_url (string)
- Do not include extra commentary, markdown, or descriptions. Just bullets of items.
"""

def encode_image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
                user_text = input("you> ").strip()
                if not user_text:
                    continue

                img_path = input("optional image path> ").strip()
                content = []
                if user_text:
                    content.append({"type": "text", "text": user_text})
                if img_path:
                    img_b64 = encode_image_to_b64(img_path)
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

                history.append({"role":"user","content":content})

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