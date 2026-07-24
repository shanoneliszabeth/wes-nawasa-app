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

LANGUAGES = ["English", "Spanish", "French", "Kreyòl", "Chinese"]

TRANSLATIONS = {
    "English": {
        "page_title": "W.E.S. - NAWASA Assist",
        "page_subtitle": "Serving {territory} · National Water & Sewerage Authority · Mode: {mode}",
        "sidebar_title": "💧 W.E.S. Settings",
        "sidebar_language": "Language",
        "sidebar_api_key_label": "Gemini API Key",
        "sidebar_api_key_help": (
            "Enter your personal Gemini API key here. "
            "This app no longer uses a shared key or Streamlit Secrets, "
            "so every visitor must provide their own key."
        ),
        "sidebar_i_am_a": "I am a...",
        "customer": "Customer",
        "field_worker": "Field Worker",
        "sidebar_select_territory": "Select Territory",
        "sidebar_clear_chat": "Clear Chat",
        "api_key_required": "Please enter your Gemini API key in the sidebar to start chatting.",
        "chat_input_placeholder": "Type your message here...",
        "field_worker_upload": "Upload a photo of a water meter or tank gauge",
        "auto_read_prompt": "Please read the meter/tank value from the photo I uploaded.",
        "thinking_spinner": "W.E.S. is thinking...",
        "field_worker_mode_note": "Note: This user is a NAWASA field worker in Field Worker mode.",
        "customer_mode_note": "Note: This user is a customer.",
        "territory_note": "Note: The user is asking specifically regarding {territory}.",
        "retry_hint": " Please retry after {retry_delay}.",
        "assistant_language_instruction": "Always answer the user's messages in English.",
        "greeting_customer": (
            "Hello! I'm W.E.S., your NAWASA assistant for {territory}. "
            "How can I help you with your water services today?"
        ),
        "greeting_field_worker": (
            "Hello! I'm W.E.S. — Field Worker mode for {territory}. "
            "Upload a photo of a water meter or tank gauge below and I'll read the value for you, "
            "or ask me anything else about NAWASA operations."
        ),
        "faq_title": "Frequently Asked Questions",
        "faq_intro": "Common NAWASA questions and answers about new service connections, billing, leaks, and disconnection.",
        "faq_apply_new_connection": "You'll need to fill out an application for a new service connection. Please review the Requirements for Private Water Service and the Terms and Conditions for Water Service before applying.",
        "faq_connection_cost": "The cost depends on your pipe size: $75 for a ½\" main, $125 for ¾\", $175 for 1\", $420 for 1¼\"–2\", or $1,000 for a 4\" main — plus variable costs such as transportation, pipes/fittings, and VAT.",
        "faq_high_consumption": "High consumption can come from estimated bills, a leak, an unsecured or easily accessible tap, or a faulty meter. To check for a leak: turn off all taps, then watch the meter dial — if it's still turning, there's a leak somewhere on the property.",
        "faq_estimated_bills": "Estimated bills are calculated using an average of your last three months' consumption.",
        "faq_disconnection": "Service may be disconnected at the customer's request, for non-payment of arrears, for wastage or abuse, or for illegal tampering with meters or fittings. The minimum threshold for disconnection due to arrears is $50, once that amount is at least 30 days overdue.",
        "quota_error": (
            "I'm getting more requests than I can handle right now (we've hit today's free usage limit). "
            "Please try again in a little while, or contact NAWASA directly at 440-2155 for immediate help. "
            "Your reference code: {ref_code}"
        ),
        "gemini_error": "Gemini request failed: {error}{retry_hint}",
    },
    "Spanish": {
        "page_title": "W.E.S. - Asistente de NAWASA",
        "page_subtitle": "Sirviendo a {territory} · Autoridad Nacional de Agua y Alcantarillado · Modo: {mode}",
        "sidebar_title": "💧 Configuración W.E.S.",
        "sidebar_language": "Idioma",
        "sidebar_api_key_label": "Clave API de Gemini",
        "sidebar_api_key_help": (
            "Ingrese su clave API personal de Gemini aquí. "
            "Esta aplicación ya no usa una clave compartida ni Streamlit Secrets, "
            "así que cada visitante debe proporcionar su propia clave."
        ),
        "sidebar_i_am_a": "Soy un...",
        "customer": "Cliente",
        "field_worker": "Trabajador de campo",
        "sidebar_select_territory": "Seleccionar territorio",
        "sidebar_clear_chat": "Borrar chat",
        "api_key_required": "Ingrese su clave API de Gemini en la barra lateral para comenzar a chatear.",
        "chat_input_placeholder": "Escribe tu mensaje aquí...",
        "field_worker_upload": "Cargue una foto de un medidor de agua o un indicador de tanque",
        "auto_read_prompt": "Por favor, lea el valor del medidor/reservorio de la foto que cargué.",
        "thinking_spinner": "W.E.S. está pensando...",
        "field_worker_mode_note": "Nota: Este usuario es un trabajador de campo de NAWASA en modo Trabajador de campo.",
        "customer_mode_note": "Nota: Este usuario es un cliente.",
        "territory_note": "Nota: El usuario está consultando específicamente sobre {territory}.",
        "retry_hint": " Por favor, vuelva a intentarlo después de {retry_delay}.",
        "assistant_language_instruction": "Responda siempre a los mensajes del usuario en español.",
        "greeting_customer": (
            "¡Hola! Soy W.E.S., tu asistente de NAWASA para {territory}. "
            "¿Cómo puedo ayudarte con tus servicios de agua hoy?"
        ),
        "greeting_field_worker": (
            "¡Hola! Estoy en modo Trabajador de campo para {territory}. "
            "Cargue una foto de un medidor de agua o un indicador de tanque a continuación y leeré el valor para usted, "
            "o pregúnteme cualquier otra cosa sobre las operaciones de NAWASA."
        ),
        "quota_error": (
            "Estoy recibiendo más solicitudes de las que puedo manejar en este momento (hemos alcanzado el límite de uso gratuito de hoy). "
            "Por favor, inténtelo de nuevo en un rato, o comuníquese con NAWASA directamente al 440-2155 para obtener ayuda inmediata. "
            "Su código de referencia: {ref_code}"
        ),
        "gemini_error": "La solicitud de Gemini falló: {error}{retry_hint}",
    },
    "French": {
        "page_title": "W.E.S. - Assistant NAWASA",
        "page_subtitle": "Au service de {territory} · Autorité Nationale de l'Eau et des Égouts · Mode : {mode}",
        "sidebar_title": "💧 Paramètres W.E.S.",
        "sidebar_language": "Langue",
        "sidebar_api_key_label": "Clé API Gemini",
        "sidebar_api_key_help": (
            "Entrez votre clé API Gemini personnelle ici. "
            "Cette application n'utilise plus de clé partagée ni de Streamlit Secrets, "
            "donc chaque visiteur doit fournir sa propre clé."
        ),
        "sidebar_i_am_a": "Je suis un...",
        "customer": "Client",
        "field_worker": "Agent de terrain",
        "sidebar_select_territory": "Sélectionner le territoire",
        "sidebar_clear_chat": "Effacer le chat",
        "api_key_required": "Veuillez entrer votre clé API Gemini dans la barre latérale pour commencer à discuter.",
        "chat_input_placeholder": "Tapez votre message ici...",
        "field_worker_upload": "Téléchargez une photo d'un compteur d'eau ou d'un indicateur de réservoir",
        "auto_read_prompt": "Veuillez lire la valeur du compteur/réservoir à partir de la photo que j'ai téléchargée.",
        "thinking_spinner": "W.E.S. réfléchit...",
        "field_worker_mode_note": "Remarque : Cet utilisateur est un agent de terrain de NAWASA en mode Agent de terrain.",
        "customer_mode_note": "Remarque : Cet utilisateur est un client.",
        "territory_note": "Remarque : L'utilisateur demande spécifiquement concernant {territory}.",
        "retry_hint": " Veuillez réessayer après {retry_delay}.",
        "assistant_language_instruction": "Répondez toujours aux messages de l'utilisateur en français.",
        "greeting_customer": (
            "Bonjour ! Je suis W.E.S., votre assistant NAWASA pour {territory}. "
            "Comment puis-je vous aider avec vos services d'eau aujourd'hui ?"
        ),
        "greeting_field_worker": (
            "Bonjour ! Je suis en mode Agent de terrain pour {territory}. "
            "Téléchargez une photo d'un compteur d'eau ou d'un indicateur de réservoir ci-dessous et je lirai la valeur pour vous, "
            "ou posez-moi toute autre question sur les opérations de NAWASA."
        ),
        "quota_error": (
            "Je reçois plus de demandes que je ne peux en traiter en ce moment (nous avons atteint la limite d'utilisation gratuite d'aujourd'hui). "
            "Veuillez réessayer dans un petit moment, ou contactez NAWASA directement au 440-2155 pour une aide immédiate. "
            "Votre code de référence : {ref_code}"
        ),
        "gemini_error": "La requête Gemini a échoué : {error}{retry_hint}",
    },
    "Kreyòl": {
        "page_title": "W.E.S. - Asistan NAWASA",
        "page_subtitle": "Sèvi {territory} · Otorite Nasyonal Dlo ak Egou · Mòd : {mode}",
        "sidebar_title": "💧 Anviwònman W.E.S.",
        "sidebar_language": "Lang",
        "sidebar_api_key_label": "Kle API Gemini",
        "sidebar_api_key_help": (
            "Antre kle API Gemini pèsonèl ou isit la. "
            "Aplikasyon sa a pa itilize kle pataje ni Streamlit Secrets ankò, "
            "kidonk chak vizitè dwe bay pwòp kle yo."
        ),
        "sidebar_i_am_a": "Mwen se yon...",
        "customer": "Kliyan",
        "field_worker": "Travayè sou teren",
        "sidebar_select_territory": "Chwazi teritwa",
        "sidebar_clear_chat": "Efase chat",
        "api_key_required": "Tanpri antre kle API Gemini ou nan ba bò a pou kòmanse chat la.",
        "chat_input_placeholder": "Ekri mesaj ou isit la...",
        "field_worker_upload": "Telechaje yon foto yon kontè dlo oswa yon endikatè rezèvwa",
        "auto_read_prompt": "Tanpri li valè kontè/rezèvwa a nan foto mwen telechaje a.",
        "thinking_spinner": "W.E.S. ap panse...",
        "field_worker_mode_note": "Remak: Itilizatè sa a se yon travayè sou teren NAWASA nan mòd Travayè sou teren.",
        "customer_mode_note": "Remak: Itilizatè sa a se yon kliyan.",
        "territory_note": "Remak: Itilizatè a ap mande espesyalman sou {territory}.",
        "retry_hint": " Tanpri re-eseye apre {retry_delay}.",
        "assistant_language_instruction": "Toujou reponn mesaj itilizatè a an Kreyòl.",
        "greeting_customer": (
            "Bonjou! Mwen se W.E.S., asistan NAWASA ou pou {territory}. "
            "Kijan mwen ka ede w ak sèvis dlo ou jodi a?"
        ),
        "greeting_field_worker": (
            "Bonjou! Mwen se W.E.S. — mòd Travayè sou teren pou {territory}. "
            "Telechaje yon foto yon kontè dlo oswa yon endikatè rezèvwa anba a e m ap li valè a pou ou, "
            "oswa mande m nenpòt lòt bagay sou operasyon NAWASA."
        ),
        "quota_error": (
            "Mwen resevwa plis demann pase sa mwen ka jere kounye a (nou frape limit itilizasyon gratis jodi a). "
            "Tanpri eseye ankò pita, oswa kontakte NAWASA dirèkteman nan 440-2155 pou asistans imedyat. "
            "Kòd referans ou : {ref_code}"
        ),
        "gemini_error": "Rekèt Gemini a echwe : {error}{retry_hint}",
    },
    "Chinese": {
        "page_title": "W.E.S. - NAWASA 助手",
        "page_subtitle": "为 {territory} 服务 · 国家水务与污水管理局 · 模式：{mode}",
        "sidebar_title": "💧 W.E.S. 设置",
        "sidebar_language": "语言",
        "sidebar_api_key_label": "Gemini API 密钥",
        "sidebar_api_key_help": (
            "在此输入您个人的 Gemini API 密钥。"
            "此应用不再使用共享密钥或 Streamlit Secrets，"
            "因此每位访客都必须提供自己的密钥。"
        ),
        "sidebar_i_am_a": "我是...",
        "customer": "客户",
        "field_worker": "现场工作人员",
        "sidebar_select_territory": "选择地区",
        "sidebar_clear_chat": "清除聊天",
        "api_key_required": "请在侧边栏中输入您的 Gemini API 密钥以开始聊天。",
        "chat_input_placeholder": "在此输入您的消息...",
        "field_worker_upload": "上传水表或水箱仪表的照片",
        "auto_read_prompt": "请读取我上传的照片中的水表/水箱读数。",
        "thinking_spinner": "W.E.S. 正在思考...",
        "field_worker_mode_note": "注意：该用户为 NAWASA 的现场工作人员，处于现场工作人员模式。",
        "customer_mode_note": "注意：该用户为客户。",
        "territory_note": "注意：用户正在询问关于 {territory} 的问题。",
        "retry_hint": " 请在 {retry_delay} 后重试。",
        "assistant_language_instruction": "始终用中文回复用户的消息。",
        "greeting_customer": (
            "您好！我是 W.E.S.，您在 {territory} 的 NAWASA 助手。"
            "我今天如何帮助您处理水务服务问题？"
        ),
        "greeting_field_worker": (
            "您好！我是 W.E.S. — {territory} 的现场工作人员模式。"
            "请上传水表或水箱仪表的照片，我会读取数值，"
            "或者您也可以询问我有关 NAWASA 运营的其他问题。"
        ),
        "quota_error": (
            "我现在收到的请求太多，无法处理（我们已达到今天的免费使用上限）。"
            "请稍后再试，或直接致电 NAWASA 440-2155 获取即时帮助。"
            "您的参考代码：{ref_code}"
        ),
        "gemini_error": "Gemini 请求失败：{error}{retry_hint}",
    },
}


def t(key: str, **kwargs) -> str:
    text = TRANSLATIONS[language].get(key, TRANSLATIONS["English"].get(key, key))
    return text.format(**kwargs)

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
- New service connections require a completed application under NAWASA's Requirements for Private Water Service and the Terms and Conditions for Water Service.
- New connection costs depend on the pipe size:
  - $75 for a ½" main
  - $125 for a ¾" main
  - $175 for a 1" main
  - $420 for a 1¼"–2" main
  - $1,000 for a 4" main
  Additional variable costs may include transportation, pipes/fittings, and VAT.
- High water consumption may be caused by an estimated bill, a leak, an unsecured or accessible tap, or a faulty meter. To check for a leak, turn off all taps and watch the meter dial; if it continues to turn, there is likely a leak.
- Estimated bills are calculated from the average of the customer's last three months' consumption.
- NAWASA may disconnect service at the customer's request, for non-payment of arrears, for wastage/abuse, or for illegal tampering with meters or fittings. The minimum disconnection threshold is $50 in arrears and at least 30 days overdue.

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
# The language selector must initialize before translations are available.
language = st.sidebar.radio(
    TRANSLATIONS["English"]["sidebar_language"],
    LANGUAGES,
    index=0,
)

st.sidebar.title(t("sidebar_title"))

# Do NOT use Streamlit Secrets or environment variables for the Gemini key.
# Every user must provide their own API key in the sidebar.
api_key_input = st.sidebar.text_input(
    t("sidebar_api_key_label"),
    value="",
    type="password",
    help=t("sidebar_api_key_help"),
)

user_mode_label = st.sidebar.radio(
    t("sidebar_i_am_a"),
    [t("customer"), t("field_worker")],
)
user_mode = "Field Worker" if user_mode_label == t("field_worker") else "Customer"
mode_label = t("field_worker") if user_mode == "Field Worker" else t("customer")

territory = st.sidebar.selectbox(
    t("sidebar_select_territory"),
    ["Grenada", "Carriacou", "Petite Martinique"],
)

if st.sidebar.button(t("sidebar_clear_chat")):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
# App Title & Layout
# ---------------------------------------------------------------------------
logo_path = "logo.png"
logo_exists = os.path.exists(logo_path)

if logo_exists:
    st.image(logo_path, width=110, output_format="PNG")

st.markdown(
    f"""
    <div class="wes-header">
        <h1>💧 {t('page_title')}</h1>
        <p>{t('page_subtitle', territory=territory, mode=mode_label)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key_input:
    st.info(t("api_key_required"), icon="🔑")
    st.stop()

# Initialize chat history (reset greeting if territory or mode changes on a fresh session)
session_key = (territory, user_mode)
if "messages" not in st.session_state or st.session_state.get("session_key") != session_key:
    greeting = (
        t("greeting_field_worker", territory=territory)
        if user_mode == "Field Worker"
        else t("greeting_customer", territory=territory)
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
        t("field_worker_upload"),
        type=["png", "jpg", "jpeg"],
    )

# ---------------------------------------------------------------------------
# Chat Input & Gemini Generation
# ---------------------------------------------------------------------------
prompt = st.chat_input(t("chat_input_placeholder"))

# Auto-trigger a reading request when an image is uploaded without extra text
if uploaded_image is not None and prompt is None:
    prompt = t("auto_read_prompt")

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
            with st.spinner(t("thinking_spinner")):
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
                        f"\n{t('field_worker_mode_note')}"
                        if user_mode == "Field Worker"
                        else f"\n{t('customer_mode_note')}"
                    )

                    language_instruction = t("assistant_language_instruction")

                    config = types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION
                        + f"\n{t('territory_note', territory=territory)}"
                        + mode_note
                        + f"\n{language_instruction}",
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
                        st.error(t("quota_error", ref_code=ref_code))
                    else:
                        retry_hint = ""
                        if hasattr(e, "error") and isinstance(e.error, dict):
                            for detail in e.error.get("details", []):
                                if isinstance(detail, dict) and detail.get("@type", "").endswith("RetryInfo"):
                                    retry_delay = detail.get("retryDelay")
                                    if retry_delay:
                                        retry_hint = t("retry_hint", retry_delay=retry_delay)
                        st.error(t("gemini_error", error=error_message, retry_hint=retry_hint))
