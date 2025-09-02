from __future__ import annotations
import asyncio, json, os, sys
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import logging
#logging.basicConfig(level=logging.DEBUG)

# Tool schema mirrored for OpenAI tool-calling
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "estyl_retrieve",
            "description": "Retrieve fashion items (single) or compose budget-constrained outfits (outfit).",
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
                    "budget": {"type": "number", "minimum": 200},
                    "limit": {"type": "integer", "minimum": 10, "maximum": 48},
                    "topk_for_rerank": {"type": "integer", "minimum": 10, "maximum": 200},
                    "offset": {"type": "integer", "minimum": 0},
                    "exclude_ids": {"type": ["array","null"], "items": {"type": "string"}},
                    "num_outfits": {"type": "integer", "minimum": 5, "maximum": 10},
                    "articles": {"type": "integer", "minimum": 5, "maximum": 5},
                    "per_cat_candidates": {"type": "integer", "minimum": 5, "maximum": 10}
                },
                "required": ["mode"]
            }
        }
    }
]


SYSTEM_PROMPT = """You are Estyl, a helpful fashion assistant.
Decide when to call the `estyl_retrieve` tool:
- Call the tool if the user asks for product suggestions, searching, filtering, budgeted looks, or outfits.
- Free-chat if the user only asks for generic fashion advice with no need for catalog retrieval.
- If the user query is vague, you can ask followup questions but make sure to keep the interaction minimum.
Return concise, helpful answers. When tool results are present, summarize clearly and include key product fields.
"""

async def run_chat():
    # Launch the MCP server via stdio
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m","estyl.mcp_server"],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            client = OpenAI()
            history = [{"role":"system","content":SYSTEM_PROMPT}]

            print("Estyl chat ready. Type your message. Ctrl+C to exit.")
            while True:
                user = input("you> ").strip()
                if not user: continue
                history.append({"role":"user","content":user})

                # Ask model; allow tool-calls
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=history,
                    tools=OPENAI_TOOLS,
                )
                msg = resp.choices[0].message
                if msg.tool_calls:
                    # Only one tool-call at a time for simplicity
                    tc = msg.tool_calls[0]
                    if tc.function.name == "estyl_retrieve":
                        args = json.loads(tc.function.arguments or "{}")

                        # Invoke MCP tool
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
                        # Unknown tool (shouldn't happen)
                        print("estyl> (unrecognized tool request)")
                else:
                    answer = msg.content
                    print(f"estyl> {answer}\n")
                    history.append({"role":"assistant","content":answer})

if __name__ == "__main__":
    asyncio.run(run_chat())