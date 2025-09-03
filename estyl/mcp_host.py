from __future__ import annotations
import asyncio, json, os, sys
from typing import Dict, Any, List, Tuple
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import logging
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

client = AsyncOpenAI()

# ------------------------- System Prompt ----------------------------
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
# --------------------------------------------------------------------

# ------------------------- Tool Schemas -----------------------------
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
# --------------------------------------------------------------------

# ------------------------- Minimal History --------------------------
# We keep ONLY the last two (user, assistant) exchanges in-memory for the LLM.
# Full raw log (including tool outputs) is appended to HISTORY_FILE as JSONL.

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def append_log(entry: Dict[str, Any]) -> None:
    # Append as JSONL for easy auditing/replay.
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"Failed to write history: {e}")

# Each element is a (user_text, assistant_text) tuple.
last_exchanges: deque[Tuple[str, str]] = deque(maxlen=MAX_EXCHANGES_IN_WINDOW)

def build_messages(user_msg: str, assistant_preview: str | None) -> List[Dict[str, Any]]:
    """
    Build the messages window: [system] + last_exchanges + current user.
    Assistant previews are short summaries of prior assistant turns.
    """
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in list(last_exchanges):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

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

                # Log user turn immediately
                append_log({"ts": now_iso(), "role": "user", "content": user})

                # Build minimal window (only last two exchanges)
                msgs = build_messages(user, None)

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
    # Run all tool calls in parallel
                    async def run_one(tc):
                        name = tc.function.name
                        args = json.loads(tc.function.arguments or "{}")
                        call_id = tc.id  # <-- needed to link tool result back
                        try:
                            tool_result = await session.call_tool(name, args)
                            out_text = tool_result.content[0].text if tool_result.content else ""
                        except Exception as e:
                            out_text = f"(tool {name} failed: {e})"
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

                # If there are tool calls, execute them ASAP and print results
                # if getattr(msg, "tool_calls", None):
                #     # Run all tool calls in parallel for speed, then print in order
                #     async def run_one(tc):
                #         name = tc.function.name
                #         args = json.loads(tc.function.arguments or "{}")
                #         result_text = ""
                #         try:
                #             tool_result = await session.call_tool(name, args)
                #             result_text = tool_result.content[0].text if tool_result.content else ""
                #         except Exception as e:
                #             result_text = f"(tool {name} failed: {e})"
                #         return {
                #             "tool_name": name,
                #             "arguments": args,
                #             "output": result_text,
                #         }

                #     tasks = [run_one(tc) for tc in msg.tool_calls]
                #     results = await asyncio.gather(*tasks, return_exceptions=False)

                #     # Print each tool’s output to the console immediately
                #     # (Assumes tools already format the list as required bullets)
                #     combined_preview_parts: List[str] = []
                #     for r in results:
                #         out = (r.get("output") or "").strip()
                #         if out:
                #             print(f"estyl> {out}\n")
                #         else:
                #             print(f"estyl> (no results)\n")

                #         # Log full tool interaction (with full output) to file
                #         append_log({
                #             "ts": now_iso(),
                #             "role": "assistant",
                #             "type": "tool_result",
                #             "tool": r.get("tool_name"),
                #             "arguments": r.get("arguments"),
                #             "output": r.get("output"),
                #         })

                #         # Keep assistant preview extremely short in memory
                #         combined_preview_parts.append(f"(used {r.get('tool_name')})")

                #     # Update minimal memory with a single compact assistant preview
                #     preview = " ".join(combined_preview_parts) if combined_preview_parts else "(tool used)"
                #     last_exchanges.append((user, preview))

                #     # Also log a compact assistant note after tools
                #     append_log({"ts": now_iso(), "role": "assistant", "content": preview})

                # else:
                #     # Plain assistant text (no tool)
                #     answer = (msg.content or "").strip() if hasattr(msg, "content") else ""
                #     print(f"estyl> {answer}\n")

                #     # Log full assistant text to file
                #     append_log({"ts": now_iso(), "role": "assistant", "content": answer})

                #     # Save a trimmed preview into minimal memory
                #     preview = answer if len(answer) <= 400 else answer[:400] + "…"
                #     last_exchanges.append((user, preview))

# ----------------------------- Main ---------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print("\nBye!")
