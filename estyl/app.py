import streamlit as st
import asyncio, json, os, sys, re
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Streamlit setup
st.set_page_config(page_title="Estyl Chat", page_icon="👗", layout="centered")
st.title("👗 Estyl - Fashion Assistant")
st.caption("Ask me for outfits, dresses, or fashion advice.")

# Tool schema
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "estyl_retrieve",
            "description": "Retrieve fashion items or outfits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["single", "outfit"]},
                    "text_query": {"type": "string"},
                    "search_with": {"type": "string", "enum": ["Text", "Image", "Text + Image"]},
                    "gender": {"type": "string", "enum": ["male","female","unisex"]},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "budget_tier": {"type": "string", "enum": ["Budget","Mid","Premium","Luxury"]},
                    "budget": {"type": "number", "minimum": 200},
                },
                "required": ["mode"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are Estyl, a helpful fashion assistant.
- Call the tool if the user asks for product suggestions or outfits.
- Free-chat if the user only asks for generic fashion advice.
- If vague, ask followups.
Return concise answers. If tool results contain links or images, include them in your summary.
"""

# Maintain history
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]

client = OpenAI()

async def mcp_chat(user_message: str):
    """Handles MCP + OpenAI interaction"""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "estyl.mcp_server"],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            st.session_state.history.append({"role": "user", "content": user_message})

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.history,
                tools=OPENAI_TOOLS,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                tc = msg.tool_calls[0]
                if tc.function.name == "estyl_retrieve":
                    args = json.loads(tc.function.arguments or "{}")

                    with st.chat_message("assistant"):
                        st.write("✨ Hold on, fetching your look...")

                    result = await session.call_tool("estyl_retrieve", args)
                    tool_output = result.content[0].text if result.content else "{}"

                    st.session_state.history.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": "estyl_retrieve",
                                "arguments": json.dumps(args)
                            }
                        }]
                    })
                    st.session_state.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": "estyl_retrieve",
                        "content": tool_output
                    })

                    final = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.history
                    )
                    answer = final.choices[0].message.content
                    st.session_state.history.append({"role": "assistant", "content": answer})
                    return answer
            else:
                answer = msg.content
                st.session_state.history.append({"role": "assistant", "content": answer})
                return answer


def render_message(role: str, content: str):
    """Render chat messages with inline images & links in a collage grid"""
    with st.chat_message(role):
        if not content:
            return

        # Find raw URLs
        urls = re.findall(r'(https?://\S+)', content)

        # Clean message text (remove URLs)
        clean_text = re.sub(r'(https?://\S+)', '', content).strip()
        if clean_text:
            st.write(clean_text)

        # Process & clean URLs
        cleaned_urls = [u.rstrip(")") for u in urls]

        # Separate images vs links
        image_urls = [
            u for u in cleaned_urls
            if u.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
            or "storage.googleapis.com" in u
        ]
        link_urls = [u for u in cleaned_urls if u not in image_urls]

        # Show images in a responsive collage
        if image_urls:
            cols = st.columns(3)  # 3 images per row
            for i, img_url in enumerate(image_urls):
                with cols[i % 3]:
                    st.image(img_url, use_container_width=True)

        # Show non-image links separately
        for url in link_urls:
            st.markdown(f"[🔗 {url}]({url})")


# Render past messages safely
for h in st.session_state.history:
    if h.get("role") in ["user", "assistant"] and "content" in h:
        render_message(h["role"], h["content"])

# Input box
if user_input := st.chat_input("Ask me for outfits or items..."):
    render_message("user", user_input)
    answer = asyncio.run(mcp_chat(user_input))
    render_message("assistant", answer)
