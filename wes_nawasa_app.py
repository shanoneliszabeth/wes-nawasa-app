"""
W.E.S. — NAWASA Assist Bot - Streamlit UI
Converted for NAWASA Grenada, Carriacou & Petite Martinique

Serves two audiences:
  - Customers: billing/leak/office/payment questions via chat
  - Field workers: snap a photo of a water meter or tank gauge and W.E.S.
    reads the digits using Gemini's vision capability (computer vision)

Run with:
    streamlit run wes_nawasa_app.py

IMPORTANT: Set your Gemini API key via the sidebar input, or by exporting
it as an environment variable before launch:
    export GEMINI_API_KEY="your-key-here"
Never hardcode API keys directly in source files.
"""

import os
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="W.E.S. - NAWASA Assist",
                   page_icon="💧", layout="centered")

# --- Deep Ocean color palette -------------------------------------------------
NAVY = "#08304A"
TEAL = "#0E7C7B"
AQUA = "#6FE0E3"
CARD = "#F4F8F9"
WHITE = "#FFFFFF"

st.markdown(
    f"""
    <style>
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: {WHITE} !important;
    }}
    section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select {{
        color: {NAVY} !important;
    }}

    /* Header banner */
    .wes-header {{
        background-color: {NAVY};
        padding: 1.1rem 1.4rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }}
    .wes-header h1 {{
        color: {WHITE} !important;
        font-size: 1.6rem;
        margin: 0;
    }}
    .wes-header p {{
        color: {AQUA} !important;
        margin: 0.2rem 0 0 0;
        font-size: 0.9rem;
    }}

    /* Chat bubbles */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {{
        background-color: {CARD};
        border-radius: 12px;
        border: 1px solid #E1EBEC;
    }}
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {{
        background-color: {TEAL};
        border-radius: 12px;
    }}
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) span {{
        color: {WHITE} !important;
    }}

    /* Buttons */
    .stButton button {{
        background-color: {TEAL};
        color: {WHITE};
        border: none;
    }}
    .stButton button:hover {{
        background-color: {NAVY};
        color: {WHITE};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

SYSTEM_INSTRUCTION = """
You are W.E.S., the official virtual assistant for the National Water & Sewerage Authority (NAWASA), serving Grenada, Carriacou, and Petite Martinique.

You serve two kinds of users:
1. Customers/households — asking about billing, leaks, outages, office locations, and payments.
2. NAWASA field workers — logging water meter or tank readings from photos.

Fact Sheet (ground truth — never state facts beyond this list):
- NAWASA provides water and sewerage services across Grenada, Carriacou, and Petite Martinique.
- Main Headquarters: The Carenage, St. George's.
- Key Sub-Offices: Grenville (St. Andrew), Gouyave (St. John), Dusty Highway (Grand Anse), and Hillsborough (Carriacou).
- Contact Hotline: Call 440-2155 or emergency 911 / 440-2155 for main line.
- Emergency Water Leaks: Direct customers to report leaks immediately via hotline or website portal.
- Payment Options: NAWASA offices, local banks (Republic Bank, Grenada Co-operative Bank), online banking, or SurePay.

Operational rules:
- You do NOT have access to any individual customer's live account balance, bill amount, or outage status — you are not connected to NAWASA's billing or operations systems. Be honest about this and direct customers to the hotline (440-2155) for account-specific or real-time outage information.
- You do NOT process payments, dispatch repair crews, or make operational decisions. You inform; NAWASA staff act.
- If a user uploads a photo of a water meter or tank gauge, read the numeric value as precisely as you can and state it clearly, along with a brief note that the reading should be confirmed against the physical meter before logging.
- Always maintain a warm, helpful, and respectful Caribbean tone.
- Stay strictly within NAWASA water/sewerage topics; politely redirect anything unrelated.
"""

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("💧 W.E.S. Settings")

# Do NOT use Streamlit Secrets or environment variables for the Gemini key.
# Every user must provide their own API key in the sidebar.
api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value="",
    type="password",
    help=(
        "Enter your personal Gemini API key here. "
        "This app no longer uses a shared key or Streamlit Secrets, "
        "so every visitor must provide their own key."
    ),
)

user_mode = st.sidebar.radio(
    "I am a...",
    ["Customer", "Field Worker"],
    help="Field Worker mode adds a photo upload for logging meter/tank readings.",
)

territory = st.sidebar.selectbox(
    "Select Territory",
    ["Grenada", "Carriacou", "Petite Martinique"]
)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
# App Title & Layout
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="wes-header">
        <h1>💧 W.E.S. — NAWASA Assist</h1>
        <p>Serving {territory} · National Water &amp; Sewerage Authority · Mode: {user_mode}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key_input:
    st.info(
        "Please enter your Gemini API key in the sidebar to start chatting.", icon="🔑")
    st.stop()

# Initialize chat history (reset greeting if territory or mode changes on a fresh session)
session_key = (territory, user_mode)
if "messages" not in st.session_state or st.session_state.get("session_key") != session_key:
    if user_mode == "Field Worker":
        greeting = (
            f"Hello! I'm W.E.S. — Field Worker mode for {territory}. "
            f"Upload a photo of a water meter or tank gauge below and I'll read the value for you, "
            f"or ask me anything else about NAWASA operations."
        )
    else:
        greeting = (
            f"Hello! I'm W.E.S., your NAWASA assistant for {territory}. "
            f"How can I help you with your water services today?"
        )
    st.session_state.messages = [{"role": "assistant", "content": greeting}]
    st.session_state.session_key = session_key

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], width=250)

# ---------------------------------------------------------------------------
# Field Worker: photo upload for meter/tank reading (computer vision)
# ---------------------------------------------------------------------------
uploaded_image = None
if user_mode == "Field Worker":
    uploaded_image = st.file_uploader(
        "Upload a photo of a water meter or tank gauge",
        type=["png", "jpg", "jpeg"],
    )

# ---------------------------------------------------------------------------
# Chat Input & Gemini Generation
# ---------------------------------------------------------------------------
prompt = st.chat_input("Type your message here...")

# Auto-trigger a reading request when an image is uploaded without extra text
if uploaded_image is not None and prompt is None:
    prompt = "Please read the meter/tank value from the photo I uploaded."

if prompt:
    image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
    image_mime = uploaded_image.type if uploaded_image is not None else None

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "image": image_bytes})
    with st.chat_message("user"):
        st.markdown(prompt)
        if image_bytes:
            st.image(image_bytes, width=250)

    with st.chat_message("assistant"):
        with st.spinner("W.E.S. is thinking..."):
            try:
                client = genai.Client(api_key=api_key_input)

                # Build full chat history for Gemini so it has conversational memory
                contents = []
                for m in st.session_state.messages:
                    role = "user" if m["role"] == "user" else "model"
                    parts = [types.Part.from_text(text=m["content"])]
                    if m.get("image") and role == "user":
                        parts.append(
                            types.Part.from_bytes(
                                data=m["image"], mime_type=image_mime or "image/jpeg")
                        )
                    contents.append(types.Content(role=role, parts=parts))

                mode_note = (
                    "\nNote: This user is a NAWASA field worker in Field Worker mode."
                    if user_mode == "Field Worker"
                    else "\nNote: This user is a customer."
                )

                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                    + f"\nNote: The user is asking specifically regarding {territory}."
                    + mode_note,
                    temperature=0.7,
                )

                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=contents,
                    config=config,
                )

                bot_reply = response.text
                st.markdown(bot_reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_reply})

            except Exception as e:
                error_message = str(e)
                if hasattr(e, "error") and isinstance(e.error, dict):
                    err = e.error
                    error_message = err.get("message", error_message)

                is_quota_error = (
                    "RESOURCE_EXHAUSTED" in error_message
                    or "quota" in error_message.lower()
                    or "limit" in error_message.lower()
                )

                if is_quota_error:
                    ref_code = st.session_state.get("ref_code", "WES-XXXXXX")
                    st.error(
                        "I'm getting more requests than I can handle right now (we've hit today's free usage limit). "
                        "Please try again in a little while, or contact NAWASA directly at 440-2155 for immediate help. "
                        f"Your reference code: {ref_code}"
                    )
                else:
                    retry_hint = ""
                    if hasattr(e, "error") and isinstance(e.error, dict):
                        for detail in e.error.get("details", []):
                            if isinstance(detail, dict) and detail.get("@type", "").endswith("RetryInfo"):
                                retry_delay = detail.get("retryDelay")
                                if retry_delay:
                                    retry_hint = f" Please retry after {retry_delay}."
                    st.error(f"Gemini request failed: {error_message}{retry_hint}")
