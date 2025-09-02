from __future__ import annotations
import asyncio, json, os, sys
from typing import Dict, Any

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import logging
logging.basicConfig(level=logging.DEBUG)

from dotenv import load_dotenv
load_dotenv()  
import aioconsole


client = OpenAI()

def load_style_yaml(path: str = "style_guide.yaml") -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

# Tool schema for function calling
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
                    "num_outfits": {"type": "integer", "minimum": 10, "maximum": 20},
                    "articles": {"type": "integer", "minimum": 5, "maximum": 7},
                    "per_cat_candidates": {"type": "integer", "minimum": 5, "maximum": 10}
                },
                "required": ["mode", "text_query", "gender"]
            }
        }
    }
]

tov = load_style_yaml()
SYSTEM_PROMPT = """You are Estyl, a friendly fashion assistant.

Decide when to call the `estyl_retrieve` tool:
- Call the tool if the user asks for product suggestions, searching, filtering, budgeted looks, or outfits.
- Free-chat if the user only asks for generic fashion advice with no need for catalog retrieval. 

Treat the STYLE GUIDE YAML below as binding rules during your entire conversation:
{tov}

## Inspiration Handling
- When the user asks for an inspiration (celebrity, city, aesthetic), always answer with 
  concrete outfit suggestions: 2–3 clothing items + optional accessory.  
- Focus on clothing and fashion accessories (bags, shoes, jewelry).  
- Do not suggest cosmetics, makeup, or beauty products.  
- Phrase it as: “For a [X]-inspired look, consider …” and list pieces with a short reason.  

## Off-Topic Handling
- If the user asks about something completely unrelated to fashion/clothing/outfits:
  • Politely refuse with a warm sentence.  
  • Redirect gently back to fashion help.  

## Other Behavior
- If user uses inappropriate or toxic language, respond with a light, style-focused redirect. Never repeat the exact same phrasing; vary between them.
- If user rejects (“too expensive / not my style”), ask 1 decisive fix question.  
- Mirror user’s language. 

"""

async def run_chat():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m","estyl.mcp_server"],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            history = [{"role":"system","content":SYSTEM_PROMPT}]

            print("Estyl chat ready. Type your message. Ctrl+C to exit.")
            while True:
                user = input("you> ").strip()
                if not user: continue
                history.append({"role":"user","content":user})

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=history,
                    tools=OPENAI_TOOLS,
                )
                msg = resp.choices[0].message
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    if tc.function.name == "estyl_retrieve":
                        args = json.loads(tc.function.arguments or "{}")

                        # Invoke MCP retrieval tool
                        result = await session.call_tool("estyl_retrieve", args)
                        tool_output = result.content[0].text if result.content else "{}"

                        # Feed tool result back to the model
                        history.append({
                            "role":"assistant",
                            "tool_calls":[{"id":tc.id,"type":"function","function":{"name":"estyl_retrieve","arguments":json.dumps(args)}}]
                        })
                        history.append({"role":"tool","tool_call_id":tc.id,"name":"estyl_retrieve","content":tool_output})

                        final = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=history
                        )
                        answer = final.choices[0].message.content
                        print(f"estyl> {answer}\n")
                        history.append({"role":"assistant","content":answer})
                    else:
                        print("estyl> (unrecognized tool request)")
                else:
                    answer = msg.content
                    print(f"estyl> {answer}\n")
                    history.append({"role":"assistant","content":answer})

if __name__ == "__main__":
    asyncio.run(run_chat())