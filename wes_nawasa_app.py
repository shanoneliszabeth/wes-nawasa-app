"""
W.E.S. — NAWASA Assist Bot - Streamlit UI
Converted for NAWASA Grenada, Carriacou & Petite Martinique

Serves two audiences:
  - Customers: billing/leak/office/payment questions via chat
  - Field workers: snap a photo of a water meter or tank gauge and W.E.S.
    reads the digits using Gemini's vision capability (computer vision)

Run with:
    streamlit run wes_nawasa_app.py

Every visitor provides their own Gemini API key in the sidebar — this app
does not use a shared key or Streamlit Secrets, so one visitor's testing
can't exhaust another visitor's quota.
"""

import os
import random
import string
import base64
import json
import re
import math
import io
from collections import Counter
from pathlib import Path
import streamlit as st
from google import genai
from google.genai import types

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# gTTS language codes for each supported app language. Kreyòl (Haitian
# Creole) has no native gTTS voice, so it falls back to French pronunciation
# as the closest approximation — not perfect, but far better than nothing.
LANG_TO_TTS_CODE = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Kreyòl": "fr",   # closest available fallback, not a true Kreyòl voice
    "Chinese": "zh-CN",
}


def synthesize_speech(text, lang_code):
    """Converts text to MP3 audio bytes via gTTS. Returns None on any
    failure (no internet, unsupported text, etc.) so a TTS hiccup never
    breaks the chat reply itself."""
    if not GTTS_AVAILABLE or not text.strip():
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang_code).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RAG-lite: ground answers in the real, verified NAWASA Q&A dataset before
# calling Gemini. Pure-Python cosine similarity over word counts — no extra
# dependencies (numpy/sklearn) required, so nothing new to pip install.
# ---------------------------------------------------------------------------
# near-exact match -> answer instantly, skip Gemini call (saves quota, guarantees accuracy)
RAG_HIGH_CONFIDENCE = 0.65
# weaker match -> still pass to Gemini as grounding context
RAG_CONTEXT_THRESHOLD = 0.20


def _tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _cosine_sim(a_tokens, b_tokens):
    a_counts, b_counts = Counter(a_tokens), Counter(b_tokens)
    common = set(a_counts) & set(b_counts)
    dot = sum(a_counts[w] * b_counts[w] for w in common)
    mag_a = math.sqrt(sum(v * v for v in a_counts.values()))
    mag_b = math.sqrt(sum(v * v for v in b_counts.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


@st.cache_data
def load_qa_dataset():
    """Loads wes_qa_dataset.json from the app folder. Returns [] if missing
    so the app still runs (just without RAG grounding) rather than crashing."""
    dataset_path = Path(__file__).resolve().parent / "wes_qa_dataset.json"
    if not dataset_path.exists():
        return []
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def retrieve_matches(query, qa_data, top_k=3):
    """Returns up to top_k (similarity, {question, answer}) pairs, sorted
    highest similarity first."""
    if not qa_data:
        return []
    q_tokens = _tokenize(query)
    scored = [(_cosine_sim(q_tokens, _tokenize(item["question"])), item)
              for item in qa_data]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# Only languages with a COMPLETE translation table below are offered.
# Adding a language to this list without a matching TRANSLATIONS entry
# used to crash the app (KeyError) — t() is now defensive against that
# too, but keep this list and TRANSLATIONS in sync as you add languages.
LANGUAGES = [
    "English",
    "Spanish",
    "French",
    "Kreyòl",
    "Chinese",
]

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
        "household_customer": "Household Customer",
        "business_customer": "Business / Commercial Customer",
        "new_connection_applicant": "New Connection Applicant",
        "field_worker": "Field Worker",
        "nawasa_staff": "NAWASA Staff",
        "sidebar_select_territory": "Select Territory",
        "sidebar_clear_chat": "Clear Chat",
        "voice_toggle_label": "🔊 Read replies aloud",
        "simple_replies_toggle": "🐢 Simple, unhurried replies",
        "api_key_required": "Please enter your Gemini API key in the sidebar to start chatting.",
        "chat_input_placeholder": "Type your message here...",
        "field_worker_upload": "Upload a photo of a water meter or tank gauge",
        "auto_read_prompt": "Please read the meter/tank value from the photo I uploaded.",
        "thinking_spinner": "W.E.S. is thinking...",
        "household_customer_mode_note": "Note: This user is a household customer. Use a warm, patient, reassuring tone, and acknowledge stress before delivering facts if their message reads as urgent (a leak, a high bill, a disconnection notice).",
        "business_customer_mode_note": "Note: This user is a business/commercial customer on NAWASA's non-domestic rate tier (higher deposit than domestic, and sewerage billed at two-thirds of the water rate instead of one-third). Use a more formal, transactional tone, and lead with continuity-focused, predictable answers — a disconnection threatens their operations, not just personal comfort.",
        "new_connection_applicant_mode_note": "Note: This user is not yet a NAWASA customer — they're asking about getting a new service connection. Explain the process from zero, assume no familiarity with NAWASA terminology, and be explicit that two different connection-fee tables exist so they don't budget off just one number.",
        "field_worker_mode_note": "Note: This user is a NAWASA field worker in Field Worker mode. Keep replies brisk and efficient — they're time-pressured, moving between properties in the field.",
        "nawasa_staff_mode_note": "Note: This user is NAWASA's own customer service staff, using W.E.S. as an internal lookup tool while a customer waits on the line. Skip conversational padding — answer with the fastest, most complete correct information, formatted for reading straight back to a customer, and include both conflicting fee tables at once if the question touches connection fees.",
        "simple_replies_instruction": "The user has asked for simple, unhurried, plain-language replies: avoid jargon, keep sentences short, confirm understanding, and be extra patient and reassuring regardless of their selected role above.",
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
        "greeting_staff": (
            "W.E.S. ready — internal lookup mode for {territory}. "
            "Ask me anything about NAWASA policy, fees, or procedures and I'll give you the fastest "
            "correct answer to read back to your customer."
        ),
        "faq_title": "Frequently Asked Questions",
        "faq_intro": "Common NAWASA questions and answers about new service connections, billing, leaks, and disconnection.",
        "faq_apply_new_connection": "You'll need to fill out an application for a new service connection. Please review the Requirements for Private Water Service and the Terms and Conditions for Water Service before applying.",
        "faq_connection_cost": "The cost depends on your pipe size: $75 for a ½\" main, $125 for ¾\", $175 for 1\", $420 for 1¼\"–2\", or $1,000 for a 4\" main — plus variable costs such as transportation, pipes/fittings, and VAT.",
        "faq_high_consumption": "High consumption can come from estimated bills, a leak, an unsecured or easily accessible tap, or a faulty meter. To check for a leak: turn off all taps, then watch the meter dial — if it's still turning, there's a leak somewhere on the property.",
        "faq_estimated_bills": "Estimated bills are calculated using an average of your last three months' consumption.",
        "faq_disconnection": "Service may be disconnected at the customer's request, for non-payment of arrears, for wastage or abuse, or for illegal tampering with meters or fittings. The minimum threshold for disconnection due to arrears is $50, once that amount is at least 30 days overdue.",
        "faq_contact": "Need help? Call 440-2155 or WhatsApp 405 5245 / 459 6064 / 405 9143. Do not share sensitive personal or payment details in chat.",
        "faq_fee_conflict": "Two published connection-fee tables exist: one from the NAWASA FAQ page and one from the 2010 gazetted regulation. Confirm with NAWASA staff before relying on either.",
        "faq_deposit_reconnection": "Deposits: Domestic $240, Non-domestic $340. Reconnection fees: Domestic $75, Non-domestic $150.",
        "faq_billing_terms": "Bills are due within 30 days of issue. Late amounts accrue 1% interest per month, and service may be discontinued after 30+ days overdue.",
        "faq_leak_check": "To self-check for a leak: turn off all taps and appliances, watch the meter. If it keeps moving, report a leak; if it stops, the issue is likely an estimated bill, an outdoor tap, or a meter problem.",
        "faq_payment_channels": "Cash: Main Office (The Carenage), Grenville, or Gouyave sub-offices. Cheque boxes: Main Office, Grenville, or Dusty Highway. Online banking: Scotia Bank, Republic Bank, Grenada Co-operative Bank, or CIBC FirstCaribbean. Payment centers: Grenada Co-operative Bank, Western Union, RBTT Bank, or River Sallee Co-operative Credit Union.",
        "faq_apply_new_connection_header": "How do I apply for a new service connection?",
        "faq_connection_header": "How much does a new service connection cost?",
        "faq_contact_header": "How can I contact NAWASA for help?",
        "faq_deposit_header": "What are deposit and reconnection fees?",
        "faq_billing_header": "What are NAWASA billing terms and late charges?",
        "faq_leak_header": "How do I check whether I have a leak?",
        "faq_high_consumption_header": "Why is my water usage high and how can I check for leaks?",
        "faq_fee_conflict_header": "What if published connection fees conflict?",
        "faq_payment_channels_header": "Where can I pay my bill?",
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
        "household_customer": "Cliente Doméstico",
        "business_customer": "Cliente Comercial / Empresarial",
        "new_connection_applicant": "Solicitante de Nueva Conexión",
        "field_worker": "Trabajador de campo",
        "nawasa_staff": "Personal de NAWASA",
        "sidebar_select_territory": "Seleccionar territorio",
        "sidebar_clear_chat": "Borrar chat",
        "voice_toggle_label": "🔊 Leer respuestas en voz alta",
        "simple_replies_toggle": "🐢 Respuestas simples y sin prisa",
        "api_key_required": "Ingrese su clave API de Gemini en la barra lateral para comenzar a chatear.",
        "chat_input_placeholder": "Escribe tu mensaje aquí...",
        "field_worker_upload": "Cargue una foto de un medidor de agua o un indicador de tanque",
        "auto_read_prompt": "Por favor, lea el valor del medidor/reservorio de la foto que cargué.",
        "thinking_spinner": "W.E.S. está pensando...",
        "household_customer_mode_note": "Nota: Este usuario es un cliente doméstico. Use un tono cálido, paciente y tranquilizador, y reconozca el estrés antes de dar los datos si el mensaje suena urgente (una fuga, una factura alta, un aviso de desconexión).",
        "business_customer_mode_note": "Nota: Este usuario es un cliente comercial/empresarial en el nivel de tarifa no doméstica de NAWASA (depósito más alto que el doméstico, y el alcantarillado se factura a dos tercios de la tarifa de agua en lugar de un tercio). Use un tono más formal y transaccional, y priorice respuestas centradas en la continuidad y previsibilidad — una desconexión amenaza sus operaciones, no solo la comodidad personal.",
        "new_connection_applicant_mode_note": "Nota: Este usuario todavía no es cliente de NAWASA — está preguntando sobre cómo obtener una nueva conexión de servicio. Explique el proceso desde cero, no asuma familiaridad con la terminología de NAWASA, y sea explícito en que existen dos tablas diferentes de tarifas de conexión para que no calcule su presupuesto con un solo número.",
        "field_worker_mode_note": "Nota: Este usuario es un trabajador de campo de NAWASA en modo Trabajador de campo. Mantenga las respuestas breves y eficientes — está bajo presión de tiempo, moviéndose entre propiedades en el campo.",
        "nawasa_staff_mode_note": "Nota: Este usuario es personal del propio servicio al cliente de NAWASA, usando W.E.S. como herramienta de consulta interna mientras un cliente espera en línea. Evite relleno conversacional — responda con la información correcta más completa y rápida, en un formato listo para leer directamente al cliente, e incluya ambas tablas de tarifas en conflicto si la pregunta trata sobre tarifas de conexión.",
        "simple_replies_instruction": "El usuario ha pedido respuestas simples, sin prisa y en lenguaje sencillo: evite la jerga, use oraciones cortas, confirme que se entendió, y sea especialmente paciente y tranquilizador, sin importar el rol seleccionado arriba.",
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
        "greeting_staff": (
            "W.E.S. listo — modo de consulta interna para {territory}. "
            "Pregúnteme lo que necesite sobre políticas, tarifas o procedimientos de NAWASA y le daré "
            "la respuesta correcta más rápida para leerla a su cliente."
        ),
        "quota_error": (
            "Estoy recibiendo más solicitudes de las que puedo manejar en este momento (hemos alcanzado el límite de uso gratuito de hoy). "
            "Por favor, inténtelo de nuevo en un rato, o comuníquese con NAWASA directamente al 440-2155 para obtener ayuda inmediata. "
            "Su código de referencia: {ref_code}"
        ),
        "gemini_error": "La solicitud de Gemini falló: {error}{retry_hint}",
        "faq_title": "Preguntas frecuentes",
        "faq_intro": "Preguntas y respuestas comunes de NAWASA sobre nuevas conexiones de servicio, facturación, fugas y desconexión.",
        "faq_apply_new_connection": "Deberá completar una solicitud de nueva conexión de servicio. Revise los Requisitos para el Servicio de Agua Privado y los Términos y Condiciones del Servicio de Agua antes de solicitar.",
        "faq_connection_cost": "El costo depende del tamaño de la tubería: $75 para ½\" principal, $125 para ¾\", $175 para 1\", $420 para 1¼\"–2\", o $1,000 para una tubería principal de 4\" — más costos variables como transporte, tuberías/accesorios y el IVA.",
        "faq_high_consumption": "El alto consumo puede deberse a facturas estimadas, una fuga, un grifo desprotegido o de fácil acceso, o un medidor defectuoso. Para revisar si hay una fuga: cierre todos los grifos y observe el dial del medidor — si sigue girando, hay una fuga en algún lugar de la propiedad.",
        "faq_estimated_bills": "Las facturas estimadas se calculan usando un promedio del consumo de sus últimos tres meses.",
        "faq_disconnection": "El servicio puede desconectarse a solicitud del cliente, por falta de pago de atrasos, por desperdicio o abuso, o por manipulación ilegal de medidores o accesorios. El umbral mínimo para la desconexión por atrasos es $50, una vez que ese monto tenga al menos 30 días de vencido.",
        "faq_contact": "¿Necesita ayuda? Llame al 440-2155 o escriba por WhatsApp al 405 5245 / 459 6064 / 405 9143. No comparta datos personales o de pago sensibles en el chat.",
        "faq_fee_conflict": "Existen dos tablas publicadas de tarifas de conexión: una de la página de preguntas frecuentes de NAWASA y otra del reglamento publicado en 2010. Confirme con el personal de NAWASA antes de basarse en cualquiera de las dos.",
        "faq_deposit_reconnection": "Depósitos: Doméstico $240, No doméstico $340. Tarifas de reconexión: Doméstico $75, No doméstico $150.",
        "faq_billing_terms": "Las facturas vencen dentro de los 30 días de su emisión. Los montos atrasados acumulan 1% de interés mensual, y el servicio puede ser suspendido después de 30 días o más de atraso.",
        "faq_leak_check": "Para autoevaluar una fuga: cierre todos los grifos y electrodomésticos, observe el medidor. Si sigue moviéndose, reporte una fuga; si se detiene, el problema probablemente sea una factura estimada, un grifo exterior o un problema del medidor.",
        "faq_payment_channels": "Efectivo: Oficina Principal (The Carenage), Grenville, o las suboficinas de Gouyave. Buzones de cheques: Oficina Principal, Grenville, o Dusty Highway. Banca en línea: Scotia Bank, Republic Bank, Grenada Co-operative Bank, o CIBC FirstCaribbean. Centros de pago: Grenada Co-operative Bank, Western Union, RBTT Bank, o River Sallee Co-operative Credit Union.",
        "faq_apply_new_connection_header": "¿Cómo solicito una nueva conexión de servicio?",
        "faq_connection_header": "¿Cuánto cuesta una nueva conexión de servicio?",
        "faq_contact_header": "¿Cómo puedo contactar a NAWASA para obtener ayuda?",
        "faq_deposit_header": "¿Cuáles son las tarifas de depósito y reconexión?",
        "faq_billing_header": "¿Cuáles son los términos de facturación y cargos por atraso de NAWASA?",
        "faq_leak_header": "¿Cómo verifico si tengo una fuga?",
        "faq_high_consumption_header": "¿Por qué es alto mi consumo de agua y cómo puedo revisar si hay fugas?",
        "faq_fee_conflict_header": "¿Qué pasa si las tarifas de conexión publicadas entran en conflicto?",
        "faq_payment_channels_header": "¿Dónde puedo pagar mi factura?",
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
        "household_customer": "Client résidentiel",
        "business_customer": "Client commercial / entreprise",
        "new_connection_applicant": "Demandeur de nouvelle connexion",
        "field_worker": "Agent de terrain",
        "nawasa_staff": "Personnel de NAWASA",
        "sidebar_select_territory": "Sélectionner le territoire",
        "sidebar_clear_chat": "Effacer le chat",
        "voice_toggle_label": "🔊 Lire les réponses à voix haute",
        "simple_replies_toggle": "🐢 Réponses simples et sans hâte",
        "api_key_required": "Veuillez entrer votre clé API Gemini dans la barre latérale pour commencer à discuter.",
        "chat_input_placeholder": "Tapez votre message ici...",
        "field_worker_upload": "Téléchargez une photo d'un compteur d'eau ou d'un indicateur de réservoir",
        "auto_read_prompt": "Veuillez lire la valeur du compteur/réservoir à partir de la photo que j'ai téléchargée.",
        "thinking_spinner": "W.E.S. réfléchit...",
        "household_customer_mode_note": "Remarque : Cet utilisateur est un client résidentiel. Utilisez un ton chaleureux, patient et rassurant, et reconnaissez le stress avant de donner les faits si le message semble urgent (une fuite, une facture élevée, un avis de déconnexion).",
        "business_customer_mode_note": "Remarque : Cet utilisateur est un client commercial/entreprise sur le tarif non domestique de NAWASA (dépôt plus élevé que domestique, et l'assainissement facturé aux deux tiers du tarif de l'eau au lieu d'un tiers). Utilisez un ton plus formel et transactionnel, et privilégiez des réponses axées sur la continuité et la prévisibilité — une déconnexion menace leurs activités, pas seulement leur confort personnel.",
        "new_connection_applicant_mode_note": "Remarque : Cet utilisateur n'est pas encore client de NAWASA — il/elle se renseigne sur l'obtention d'une nouvelle connexion de service. Expliquez le processus depuis le début, ne présumez aucune familiarité avec la terminologie de NAWASA, et précisez explicitement qu'il existe deux tableaux de frais de connexion différents afin qu'il/elle ne budgétise pas sur un seul chiffre.",
        "field_worker_mode_note": "Remarque : Cet utilisateur est un agent de terrain de NAWASA en mode Agent de terrain. Gardez les réponses brèves et efficaces — il/elle est pressé(e) par le temps, se déplaçant entre les propriétés sur le terrain.",
        "nawasa_staff_mode_note": "Remarque : Cet utilisateur fait partie du personnel du service client de NAWASA, utilisant W.E.S. comme outil de consultation interne pendant qu'un client attend en ligne. Évitez le remplissage conversationnel — répondez avec l'information correcte la plus complète et la plus rapide, formatée pour être lue directement à un client, et incluez les deux tableaux de frais en conflit si la question porte sur les frais de connexion.",
        "simple_replies_instruction": "L'utilisateur a demandé des réponses simples, sans hâte, en langage clair : évitez le jargon, gardez des phrases courtes, confirmez la compréhension, et soyez particulièrement patient et rassurant, quel que soit le rôle sélectionné ci-dessus.",
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
        "greeting_staff": (
            "W.E.S. prêt — mode de consultation interne pour {territory}. "
            "Posez-moi toute question sur les politiques, frais ou procédures de NAWASA et je vous "
            "donnerai la réponse correcte la plus rapide à lire à votre client."
        ),
        "quota_error": (
            "Je reçois plus de demandes que je ne peux en traiter en ce moment (nous avons atteint la limite d'utilisation gratuite d'aujourd'hui). "
            "Veuillez réessayer dans un petit moment, ou contactez NAWASA directement au 440-2155 pour une aide immédiate. "
            "Votre code de référence : {ref_code}"
        ),
        "gemini_error": "La requête Gemini a échoué : {error}{retry_hint}",
        "faq_title": "Questions fréquemment posées",
        "faq_intro": "Questions et réponses courantes de NAWASA sur les nouvelles connexions de service, la facturation, les fuites et la déconnexion.",
        "faq_apply_new_connection": "Vous devrez remplir une demande de nouvelle connexion de service. Veuillez consulter les Exigences pour le service d'eau privé et les Termes et conditions du service d'eau avant de faire votre demande.",
        "faq_connection_cost": "Le coût dépend de la taille de votre tuyau : 75 $ pour ½\" principal, 125 $ pour ¾\", 175 $ pour 1\", 420 $ pour 1¼\"–2\", ou 1 000 $ pour un tuyau principal de 4\" — plus des coûts variables comme le transport, les tuyaux/raccords et la TVA.",
        "faq_high_consumption": "Une consommation élevée peut provenir de factures estimées, d'une fuite, d'un robinet non sécurisé ou facilement accessible, ou d'un compteur défectueux. Pour vérifier une fuite : fermez tous les robinets, puis observez le cadran du compteur — s'il tourne toujours, il y a une fuite quelque part sur la propriété.",
        "faq_estimated_bills": "Les factures estimées sont calculées à partir d'une moyenne de votre consommation des trois derniers mois.",
        "faq_disconnection": "Le service peut être déconnecté à la demande du client, pour non-paiement d'arriérés, pour gaspillage ou abus, ou pour altération illégale des compteurs ou raccords. Le seuil minimum de déconnexion pour arriérés est de 50 $, une fois que ce montant est en retard d'au moins 30 jours.",
        "faq_contact": "Besoin d'aide ? Appelez le 440-2155 ou WhatsApp 405 5245 / 459 6064 / 405 9143. Ne partagez pas de données personnelles ou de paiement sensibles dans le chat.",
        "faq_fee_conflict": "Deux tableaux de frais de connexion publiés existent : l'un de la page FAQ de NAWASA et l'autre du règlement publié en 2010. Confirmez avec le personnel de NAWASA avant de vous fier à l'un ou l'autre.",
        "faq_deposit_reconnection": "Dépôts : Domestique 240 $, Non domestique 340 $. Frais de reconnexion : Domestique 75 $, Non domestique 150 $.",
        "faq_billing_terms": "Les factures sont dues dans les 30 jours suivant leur émission. Les montants en retard accumulent 1 % d'intérêt par mois, et le service peut être interrompu après 30 jours ou plus de retard.",
        "faq_leak_check": "Pour vérifier vous-même une fuite : fermez tous les robinets et appareils, observez le compteur. S'il continue de bouger, signalez une fuite ; s'il s'arrête, le problème est probablement une facture estimée, un robinet extérieur, ou un problème de compteur.",
        "faq_payment_channels": "Espèces : Bureau principal (The Carenage), Grenville, ou les sous-bureaux de Gouyave. Boîtes à chèques : Bureau principal, Grenville, ou Dusty Highway. Services bancaires en ligne : Scotia Bank, Republic Bank, Grenada Co-operative Bank, ou CIBC FirstCaribbean. Centres de paiement : Grenada Co-operative Bank, Western Union, RBTT Bank, ou River Sallee Co-operative Credit Union.",
        "faq_apply_new_connection_header": "Comment puis-je demander une nouvelle connexion de service ?",
        "faq_connection_header": "Combien coûte une nouvelle connexion de service ?",
        "faq_contact_header": "Comment puis-je contacter NAWASA pour obtenir de l'aide ?",
        "faq_deposit_header": "Quels sont les frais de dépôt et de reconnexion ?",
        "faq_billing_header": "Quelles sont les conditions de facturation et les frais de retard de NAWASA ?",
        "faq_leak_header": "Comment vérifier si j'ai une fuite ?",
        "faq_high_consumption_header": "Pourquoi ma consommation d'eau est-elle élevée et comment vérifier les fuites ?",
        "faq_fee_conflict_header": "Que faire si les frais de connexion publiés entrent en conflit ?",
        "faq_payment_channels_header": "Où puis-je payer ma facture ?",
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
        "household_customer": "Kliyan Kay",
        "business_customer": "Kliyan Biznis / Komèsyal",
        "new_connection_applicant": "Aplikan pou Nouvo Koneksyon",
        "field_worker": "Travayè sou teren",
        "nawasa_staff": "Anplwaye NAWASA",
        "sidebar_select_territory": "Chwazi teritwa",
        "sidebar_clear_chat": "Efase chat",
        "voice_toggle_label": "🔊 Li repons yo awotvwa",
        "simple_replies_toggle": "🐢 Repons senp, san prese",
        "api_key_required": "Tanpri antre kle API Gemini ou nan ba bò a pou kòmanse chat la.",
        "chat_input_placeholder": "Ekri mesaj ou isit la...",
        "field_worker_upload": "Telechaje yon foto yon kontè dlo oswa yon endikatè rezèvwa",
        "auto_read_prompt": "Tanpri li valè kontè/rezèvwa a nan foto mwen telechaje a.",
        "thinking_spinner": "W.E.S. ap panse...",
        "household_customer_mode_note": "Remak: Itilizatè sa a se yon kliyan kay. Sèvi ak yon ton cho, pasyan, e ki rasire, epi rekonèt estrès anvan ou bay enfòmasyon si mesaj la sanble ijan (yon fwit, yon gwo bòdwo, yon avi dekoneksyon).",
        "business_customer_mode_note": "Remak: Itilizatè sa a se yon kliyan biznis/komèsyal sou nivo tarif non-domestik NAWASA a (depo pi wo pase domestik, e egou yo chaje de tyè tarif dlo a olye de yon tyè). Sèvi ak yon ton pi fòmèl, transaksyonèl, epi mete devan repons ki konsantre sou kontinwite ak previzibilite — yon dekoneksyon menase operasyon yo, pa sèlman konfò pèsonèl.",
        "new_connection_applicant_mode_note": "Remak: Itilizatè sa a poko yon kliyan NAWASA — l ap mande sou kijan pou jwenn yon nouvo koneksyon sèvis. Eksplike pwosesis la soti nan zewo, pa sipoze li konnen tèminoloji NAWASA, epi di klèman gen de tablo frè koneksyon diferan pou li pa fè bidjè l sou yon sèl chif.",
        "field_worker_mode_note": "Remak: Itilizatè sa a se yon travayè sou teren NAWASA nan mòd Travayè sou teren. Kenbe repons yo kout e efikas — li anba presyon tan, l ap deplase ant pwopriyete nan teren an.",
        "nawasa_staff_mode_note": "Remak: Itilizatè sa a se pwòp anplwaye sèvis kliyan NAWASA, k ap itilize W.E.S. kòm yon zouti rechèch entèn pandan yon kliyan ap tann sou liy lan. Evite ranbi konvèsasyon — reponn ak enfòmasyon ki kòrèk, pi konplè e pi rapid, fòmate pou li dirèkteman bay yon kliyan, epi mete tou de tablo frè ki an konfli si kesyon an konsène frè koneksyon.",
        "simple_replies_instruction": "Itilizatè a mande repons senp, san prese, an lang klè: evite jagon, kenbe fraz yo kout, konfime konpreyansyon, epi montre plis pasyans ak rasirans, kèlkeswa wòl yo chwazi anwo a.",
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
        "greeting_staff": (
            "W.E.S. pare — mòd rechèch entèn pou {territory}. "
            "Mande m nenpòt bagay sou politik, frè, oswa pwosedi NAWASA e m ap ba ou repons kòrèk "
            "ki pi rapid pou li bay kliyan ou an."
        ),
        "quota_error": (
            "Mwen resevwa plis demann pase sa mwen ka jere kounye a (nou frape limit itilizasyon gratis jodi a). "
            "Tanpri eseye ankò pita, oswa kontakte NAWASA dirèkteman nan 440-2155 pou asistans imedyat. "
            "Kòd referans ou : {ref_code}"
        ),
        "gemini_error": "Rekèt Gemini a echwe : {error}{retry_hint}",
        "faq_title": "Kesyon yo poze souvan",
        "faq_intro": "Kesyon ak repons NAWASA komen sou nouvo koneksyon sèvis, fakti, fwit dlo, ak dekoneksyon.",
        "faq_apply_new_connection": "Ou pral bezwen ranpli yon aplikasyon pou yon nouvo koneksyon sèvis. Tanpri revize Kondisyon pou Sèvis Dlo Prive a ak Tèm ak Kondisyon pou Sèvis Dlo a anvan ou aplike.",
        "faq_connection_cost": "Pri a depann sou gwosè tiyo ou a: $75 pou yon prensipal ½\", $125 pou ¾\", $175 pou 1\", $420 pou 1¼\"–2\", oswa $1,000 pou yon prensipal 4\" — plis frè varyab tankou transpò, tiyo/akseswa, ak TVA.",
        "faq_high_consumption": "Gwo konsomasyon ka soti nan bòdwo estime, yon fwit dlo, yon wobinè ki pa sekirize oswa ki fasil pou jwenn, oswa yon kontè ki gen pwoblèm. Pou tcheke si gen yon fwit: fèmen tout wobinè yo, epi gade kadran kontè a — si li toujou ap vire, gen yon fwit yon kote sou pwopriyete a.",
        "faq_estimated_bills": "Bòdwo estime yo kalkile lè yo itilize yon mwayèn konsomasyon twa dènye mwa ou yo.",
        "faq_disconnection": "Sèvis la ka dekonekte sou demann kliyan an, pou non-peman aryere, pou gaspiyaj oswa abi, oswa pou manipilasyon ilegal kontè oswa akseswa yo. Sèy minimòm pou dekoneksyon poutèt aryere se $50, yon fwa montan sa a gen omwen 30 jou an reta.",
        "faq_contact": "Ou bezwen èd? Rele 440-2155 oswa WhatsApp 405 5245 / 459 6064 / 405 9143. Pa pataje detay pèsonèl oswa peman sansib nan chat la.",
        "faq_fee_conflict": "Gen de tablo frè koneksyon ki pibliye: youn soti nan paj FAQ NAWASA a ak youn soti nan règleman 2010 ki pibliye ofisyèlman an. Konfime ak anplwaye NAWASA anvan ou fè konfyans nan youn ladan yo.",
        "faq_deposit_reconnection": "Depo: Domestik $240, Non-domestik $340. Frè rekoneksyon: Domestik $75, Non-domestik $150.",
        "faq_billing_terms": "Bòdwo yo dwe peye nan 30 jou apre yo pibliye yo. Montan an reta akimile 1% enterè pa mwa, epi sèvis la ka dekonekte apre 30+ jou an reta.",
        "faq_leak_check": "Pou tcheke tèt ou pou yon fwit: fèmen tout wobinè ak aparèy, gade kontè a. Si li kontinye ap deplase, rapòte yon fwit; si li rete, pwoblèm nan gen plis chans se yon bòdwo estime, yon wobinè deyò, oswa yon pwoblèm kontè.",
        "faq_payment_channels": "Kach: Biwo Prensipal (The Carenage), Grenville, oswa sou-biwo Gouyave yo. Bwat chèk: Biwo Prensipal, Grenville, oswa Dusty Highway. Bank an liy: Scotia Bank, Republic Bank, Grenada Co-operative Bank, oswa CIBC FirstCaribbean. Sant peman: Grenada Co-operative Bank, Western Union, RBTT Bank, oswa River Sallee Co-operative Credit Union.",
        "faq_apply_new_connection_header": "Kijan mwen aplike pou yon nouvo koneksyon sèvis?",
        "faq_connection_header": "Konbyen yon nouvo koneksyon sèvis koute?",
        "faq_contact_header": "Kijan mwen ka kontakte NAWASA pou èd?",
        "faq_deposit_header": "Ki frè depo ak rekoneksyon yo?",
        "faq_billing_header": "Ki tèm fakti NAWASA ak chaj reta yo?",
        "faq_leak_header": "Kijan mwen tcheke si mwen gen yon fwit dlo?",
        "faq_high_consumption_header": "Poukisa itilizasyon dlo mwen wo e kijan mwen ka tcheke pou fwit?",
        "faq_fee_conflict_header": "Kisa ki rive si frè koneksyon ki pibliye yo an konfli?",
        "faq_payment_channels_header": "Kote mwen ka peye bòdwo mwen?",
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
        "household_customer": "住宅客户",
        "business_customer": "商业/企业客户",
        "new_connection_applicant": "新接通申请人",
        "field_worker": "现场工作人员",
        "nawasa_staff": "NAWASA 员工",
        "sidebar_select_territory": "选择地区",
        "sidebar_clear_chat": "清除聊天",
        "voice_toggle_label": "🔊 朗读回复",
        "simple_replies_toggle": "🐢 简单、从容的回复",
        "api_key_required": "请在侧边栏中输入您的 Gemini API 密钥以开始聊天。",
        "chat_input_placeholder": "在此输入您的消息...",
        "field_worker_upload": "上传水表或水箱仪表的照片",
        "auto_read_prompt": "请读取我上传的照片中的水表/水箱读数。",
        "thinking_spinner": "W.E.S. 正在思考...",
        "household_customer_mode_note": "注意：该用户是住宅客户。请使用温暖、耐心、令人安心的语气，如果信息显得紧急（漏水、高额账单、断供通知），请先安抚情绪，再给出事实。",
        "business_customer_mode_note": "注意：该用户是使用 NAWASA 非住宅费率的商业/企业客户（押金高于住宅客户，且污水费按水费的三分之二而非三分之一计算）。请使用更正式、事务性的语气，优先给出以业务连续性和可预测性为核心的答复——断供威胁的是他们的运营，而不仅仅是个人生活的便利。",
        "new_connection_applicant_mode_note": "注意：该用户还不是 NAWASA 的客户——他们在询问如何办理新的服务接通。请从零开始解释整个流程，不要假设对方熟悉 NAWASA 的术语，并明确说明存在两份不同的接通费用表，以免对方只按一个数字做预算。",
        "field_worker_mode_note": "注意：该用户为 NAWASA 的现场工作人员，处于现场工作人员模式。请保持回复简洁高效——他们时间紧张，需要在多个物业之间往返。",
        "nawasa_staff_mode_note": "注意：该用户是 NAWASA 自己的客服人员，在客户于线上等待时，将 W.E.S. 用作内部查询工具。请省去寒暄——直接给出最快、最完整、正确的信息，格式便于直接读给客户听，如果问题涉及接通费用，请同时列出两份互相冲突的费用表。",
        "simple_replies_instruction": "用户要求简单、从容、通俗易懂的回复：避免使用术语，句子保持简短，确认对方理解，并且无论上面选择了哪种角色，都要格外耐心、令人安心。",
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
        "greeting_staff": (
            "W.E.S. 已就绪 — {territory} 内部查询模式。"
            "请随时向我询问 NAWASA 的政策、费用或流程问题，我会给您最快、最正确的答案，"
            "方便您直接读给客户听。"
        ),
        "quota_error": (
            "我现在收到的请求太多，无法处理（我们已达到今天的免费使用上限）。"
            "请稍后再试，或直接致电 NAWASA 440-2155 获取即时帮助。"
            "您的参考代码：{ref_code}"
        ),
        "gemini_error": "Gemini 请求失败：{error}{retry_hint}",
        "faq_title": "常见问题",
        "faq_intro": "关于新服务接通、账单、漏水和断供的常见 NAWASA 问答。",
        "faq_apply_new_connection": "您需要填写新服务接通申请表。申请前请查阅《私人供水服务要求》和《供水服务条款与条件》。",
        "faq_connection_cost": "费用取决于管道尺寸：½\" 主管 $75，¾\" 为 $125，1\" 为 $175，1¼\"–2\" 为 $420，4\" 主管为 $1,000 — 另加运输、管道/配件及增值税等可变费用。",
        "faq_high_consumption": "用水量高可能来自估算账单、漏水、未加保护或容易触及的水龙头，或水表故障。检查漏水的方法：关闭所有水龙头，然后观察水表指针 —— 如果仍在转动，说明物业内某处存在漏水。",
        "faq_estimated_bills": "估算账单是根据您最近三个月用水量的平均值计算的。",
        "faq_disconnection": "服务可能因客户要求、欠款未付、浪费或滥用，或非法改装水表或配件而被断供。因欠款断供的最低门槛为 $50，且该欠款已逾期至少 30 天。",
        "faq_contact": "需要帮助？请致电 440-2155 或通过 WhatsApp 联系 405 5245 / 459 6064 / 405 9143。请勿在聊天中分享敏感的个人或付款信息。",
        "faq_fee_conflict": "目前存在两份已公布的接通费用表：一份来自 NAWASA 常见问题页面，另一份来自 2010 年公报法规。使用前请与 NAWASA 工作人员确认。",
        "faq_deposit_reconnection": "押金：住宅 $240，非住宅 $340。复通费：住宅 $75，非住宅 $150。",
        "faq_billing_terms": "账单须在开具后 30 天内支付。逾期金额每月累积 1% 利息，逾期 30 天以上服务可能被中断。",
        "faq_leak_check": "自行检查漏水的方法：关闭所有水龙头和用水电器，观察水表。如果水表仍在转动，请报告漏水；如果水表停止转动，问题更可能是估算账单、室外水龙头或水表故障。",
        "faq_payment_channels": "现金：主办公室（The Carenage）、Grenville 或 Gouyave 分办公室。支票箱：主办公室、Grenville 或 Dusty Highway。网上银行：Scotia Bank、Republic Bank、Grenada Co-operative Bank 或 CIBC FirstCaribbean。付款中心：Grenada Co-operative Bank、Western Union、RBTT Bank 或 River Sallee Co-operative Credit Union。",
        "faq_apply_new_connection_header": "如何申请新的服务接通？",
        "faq_connection_header": "新服务接通需要多少费用？",
        "faq_contact_header": "如何联系 NAWASA 获取帮助？",
        "faq_deposit_header": "押金和复通费是多少？",
        "faq_billing_header": "NAWASA 的账单条款和逾期费用是什么？",
        "faq_leak_header": "如何检查是否漏水？",
        "faq_high_consumption_header": "为什么我的用水量偏高？如何检查漏水？",
        "faq_fee_conflict_header": "如果公布的接通费用相互冲突怎么办？",
        "faq_payment_channels_header": "我可以在哪里缴纳账单？",
    },
}


language = "English"


def t(key: str, **kwargs) -> str:
    # FIX (bug #1): defensive against a language in LANGUAGES that has no
    # matching TRANSLATIONS entry — falls back to English instead of a
    # hard KeyError crash. Also guards the inner key lookup the same way.
    selected_language = globals().get("language", "English")
    if not isinstance(selected_language, str):
        selected_language = "English"
    lang_table = TRANSLATIONS.get(selected_language, TRANSLATIONS["English"])
    text = lang_table.get(key, TRANSLATIONS["English"].get(key, key))
    return text.format(**kwargs)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
app_dir = Path(__file__).resolve().parent
logo_path = app_dir / "logo.png"
logo_base64 = ""
if logo_path.exists():
    logo_base64 = base64.b64encode(logo_path.read_bytes()).decode()

st.set_page_config(page_title="W.E.S. - NAWASA Assist",
                   page_icon="💧", layout="centered")

assistant_avatar_style = (
    f"""
    div[data-testid="chatAvatarIcon-assistant"] > div {{
        background: transparent !important;
        background-image: url("data:image/png;base64,{logo_base64}") !important;
        background-size: contain !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        border-radius: 50% !important;
    }}
    div[data-testid="chatAvatarIcon-assistant"] svg {{
        display: none !important;
    }}
    """
    if logo_base64 else ""
)

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
    }}    {assistant_avatar_style}    </style>
    """,
    unsafe_allow_html=True,
)

# FIX (bug #3): restored the real, live-verified payment channels instead
# of the generic placeholder list, and removed "SurePay" — it was never
# confirmed on NAWASA's actual site and shouldn't be stated as fact.
# FIX (bug #2): {ref_code} stays as a template placeholder here; it is
# filled in via .format(ref_code=...) at call time, below, instead of
# being silently concatenated as literal, unformatted text.
SYSTEM_INSTRUCTION_TEMPLATE = """
You are W.E.S., the official virtual assistant for the National Water & Sewerage Authority (NAWASA), serving Grenada, Carriacou, and Petite Martinique.

You serve two kinds of users:
1. Customers/households — asking about billing, leaks, outages, office locations, and payments.
2. NAWASA field workers — logging water meter or tank readings from photos.

Fact Sheet (ground truth — never state facts beyond this list):
- NAWASA provides water and sewerage services across Grenada, Carriacou, and Petite Martinique.
- Main Headquarters: The Carenage, St. George's.
- Key Sub-Offices: Grenville (St. Andrew), Gouyave (St. John), Dusty Highway (Grand Anse), and Hillsborough (Carriacou).
- Contact Hotline: (473) 440-2155. WhatsApp: 405 5245 / 459 6064 / 405 9143. Emergency: 911 or 440-2155.
- Emergency Water Leaks: Direct customers to report leaks immediately via the hotline, WhatsApp, or the website portal.
- Bill Payment Channels: Cash at Main Office (The Carenage), Grenville, or Gouyave sub-offices; cheque boxes at Main Office, Grenville, or Dusty Highway; online banking via Scotia Bank, Republic Bank, Grenada Co-operative Bank, or CIBC FirstCaribbean; payment centers at Grenada Co-operative Bank, Western Union, RBTT Bank, or River Sallee Co-operative Credit Union.
- New Connection Fees: TWO different published figures exist and have NOT been reconciled — always present both and recommend confirming with the hotline: (a) per the NAWASA FAQ page: ½"=$75, ¾"=$125, 1"=$175, 1¼"–2"=$420, 4"=$1,000; (b) per the official 2010 gazetted regulation: ½"=$80, ¾"=$120, 1"=$175, 1½"–2"=$420, 2½"–3"=$1,000, 4"=$1,200, over 4"=$1,500. There is also an unverified 2024 news report suggesting a revised service charge of EC$340–$8,000 for new/reconnecting customers — mention this exists but is NOT confirmed, and must be checked with staff.
- Service Charge (deposit, per 2010 regulation): Domestic buildings $240; Non-domestic $340. (Note: possibly superseded — see above.)
- Reconnection Fee (per 2010 regulation): Domestic $75; Non-domestic $150.
- Billing Terms: Bills are due within 30 days of issue. Amounts unpaid past 30 days accrue 1% interest per month, and NAWASA may discontinue service without further notice once 30+ days overdue.
- Disconnection Threshold (per FAQ): minimum $50 in arrears, at least 30 days overdue.
- Estimated Bills: calculated using the average of the customer's last three months' consumption.
- Sewerage Rate Formula: Domestic = one third of the monthly water rate; Non-domestic = two thirds of the monthly water rate.

Leak self-diagnosis (walk the customer through this instead of just deflecting to the hotline):
1. Ask them to turn off every tap and water-using appliance in the property.
2. Ask them to watch the water meter dial.
3. If the dial is still turning with everything off, that confirms a leak — tell them to report it via the hotline, WhatsApp, or the website portal.
4. If the dial is still, the high bill is more likely an estimated bill, an unsecured/accessible outdoor tap, or a meter issue — suggest they contact the hotline to investigate further.

Operational rules:
- You do NOT have access to any individual customer's live account balance, bill amount, or real-time outage status — you are not connected to NAWASA's billing or operations systems. Be honest about this and direct customers to the hotline for account-specific or real-time information.
- You do NOT process payments, dispatch repair crews, or make operational decisions. You inform; NAWASA staff act.
- When published facts conflict (like the two connection-fee tables above), state both clearly with their source rather than picking one — never silently average or guess which is current.
- If a user uploads a photo of a water meter or tank gauge, read the numeric value as precisely as you can and state it clearly, along with a brief note that the reading should be confirmed against the physical meter before logging.
- Always maintain a warm, helpful, and respectful Caribbean tone.
- Stay strictly within NAWASA water/sewerage topics; politely redirect anything unrelated.

Escalation:
- When a request needs a human (an account-specific question, a complaint, an emergency, or anything outside what you can confidently answer), soften the handoff, tell the user to contact the hotline ((473) 440-2155) or WhatsApp (405 5245 / 459 6064 / 405 9143), and give them their session reference code so staff can find their conversation: {ref_code}
"""

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
# Use a hidden expander so the sidebar only shows the language label until clicked.
with st.sidebar.expander(TRANSLATIONS["English"]["sidebar_language"], expanded=False):
    language = st.selectbox("", LANGUAGES, index=0)

st.sidebar.title(t("sidebar_title"))

# FIX (bug #2, part A): generate one reference code per browser session.
# This was missing entirely in the pasted version — st.session_state.ref_code
# was referenced in the quota-error fallback but never actually created,
# so it always silently showed the placeholder "WES-XXXXXX".
if "ref_code" not in st.session_state:
    st.session_state.ref_code = "WES-" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )

# Every visitor provides their own Gemini API key — no shared key or
# Streamlit Secrets fallback, so one visitor's testing can't burn through
# another visitor's daily quota.
api_key_input = st.sidebar.text_input(
    t("sidebar_api_key_label"),
    value="",
    type="password",
    help=t("sidebar_api_key_help"),
)

ROLE_KEYS = [
    "household_customer",
    "business_customer",
    "new_connection_applicant",
    "field_worker",
    "nawasa_staff",
]
role_labels = [t(role_key) for role_key in ROLE_KEYS]
selected_role_label = st.sidebar.radio(t("sidebar_i_am_a"), role_labels)
user_mode = ROLE_KEYS[role_labels.index(selected_role_label)]
mode_label = selected_role_label

territory = st.sidebar.selectbox(
    t("sidebar_select_territory"),
    ["Grenada", "Carriacou", "Petite Martinique"],
)

if st.sidebar.button(t("sidebar_clear_chat")):
    st.session_state.messages = []
    st.rerun()

voice_enabled = st.sidebar.checkbox(t("voice_toggle_label"), value=False)
if voice_enabled and not GTTS_AVAILABLE:
    st.sidebar.warning(
        "gTTS isn't installed — add `gTTS` to requirements.txt to enable voice replies.")

simple_replies_enabled = st.sidebar.checkbox(
    t("simple_replies_toggle"), value=False)

st.sidebar.caption("☎ Hotline: 440-2155")
st.sidebar.caption("📱 WhatsApp: 405 5245 / 459 6064 / 405 9143")
st.sidebar.caption(
    f"🔖 Your reference code: **{st.session_state.ref_code}** — quote this if transferred to NAWASA staff.")
st.sidebar.caption(
    "ℹ️ Messages and photos are processed by Google's Gemini API (cloud-based, outside Grenada). Do not share sensitive personal or payment details in chat.")

# ---------------------------------------------------------------------------
# App Title & Layout
# ---------------------------------------------------------------------------
logo_exists = logo_path.exists()

if logo_exists:
    st.markdown(
        f"""
        <div style="width:100%; display:flex; justify-content:center; align-items:center; margin-bottom:1rem;">
            <img src="data:image/png;base64,{logo_base64}" width="110" style="display:block; margin:0 auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="wes-header">
        <h1>💧 {t('page_title')}</h1>
        <p>{t('page_subtitle', territory=territory, mode=mode_label)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander(t("faq_title"), expanded=True):
    st.markdown(f"<p>{t('faq_intro')}</p>", unsafe_allow_html=True)

    with st.expander(t("faq_contact_header")):
        st.write(t("faq_contact"))

    with st.expander(t("faq_apply_new_connection_header")):
        st.write(t("faq_apply_new_connection"))

    with st.expander(t("faq_connection_header")):
        st.write(t("faq_connection_cost"))

    with st.expander(t("faq_fee_conflict_header")):
        st.write(t("faq_fee_conflict"))

    with st.expander(t("faq_deposit_header")):
        st.write(t("faq_deposit_reconnection"))

    with st.expander(t("faq_billing_header")):
        st.write(t("faq_billing_terms"))

    with st.expander(t("faq_high_consumption_header")):
        st.write(t("faq_high_consumption"))

    with st.expander(t("faq_leak_header")):
        st.write(t("faq_leak_check"))

    with st.expander(t("faq_payment_channels_header")):
        st.write(t("faq_payment_channels"))

if not api_key_input:
    st.info(t("api_key_required"), icon="🔑")
    st.stop()

# Initialize chat history (reset greeting if territory or mode changes on a fresh session)
session_key = (territory, user_mode, language)
if "messages" not in st.session_state or st.session_state.get("session_key") != session_key:
    if user_mode == "field_worker":
        greeting = t("greeting_field_worker", territory=territory)
    elif user_mode == "nawasa_staff":
        greeting = t("greeting_staff", territory=territory)
    else:
        greeting = t("greeting_customer", territory=territory)
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
if user_mode == "field_worker":
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

    # RAG-lite: check the verified Q&A dataset before touching the Gemini API.
    # A photo attached means this is a vision task Gemini alone can do, so
    # retrieval is skipped in that case.
    qa_data = load_qa_dataset()
    rag_matches = [] if image_bytes else retrieve_matches(
        prompt, qa_data, top_k=3)
    top_sim, top_item = (rag_matches[0] if rag_matches else (0.0, None))

    with st.chat_message("assistant"):
        if top_sim >= RAG_HIGH_CONFIDENCE:
            # Near-exact match to a verified FAQ — answer instantly, no API
            # call needed. Faster, free, and guaranteed accurate.
            bot_reply = top_item["answer"] + \
                "\n\n*(Answered instantly from NAWASA's verified FAQ.)*"
            st.markdown(bot_reply)
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_reply})
            if voice_enabled:
                audio_bytes = synthesize_speech(
                    top_item["answer"], LANG_TO_TTS_CODE.get(language, "en"))
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
        else:
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

                    mode_note = f"\n{t(user_mode + '_mode_note')}"

                    language_instruction = t("assistant_language_instruction")

                    # FIX (bug #2, part B): .format(ref_code=...) actually fills
                    # in the real per-session code now, instead of the literal
                    # "{ref_code}" text leaking straight into Gemini's instructions.
                    filled_system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
                        ref_code=st.session_state.ref_code
                    )

                    # RAG grounding: pass any decent (but not instant-confidence)
                    # matches from the verified dataset as reference context,
                    # so Gemini's answer stays anchored to real facts instead
                    # of only what's baked into the static fact sheet.
                    rag_context = ""
                    relevant = [(s, i)
                                for s, i in rag_matches if s >= RAG_CONTEXT_THRESHOLD]
                    if relevant:
                        lines = "\n".join(
                            f'- Q: {i["question"]}\n  A: {i["answer"]}' for _, i in relevant
                        )
                        rag_context = (
                            "\n\nRetrieved reference facts from NAWASA's verified FAQ dataset "
                            "(use these if relevant to the user's question; ignore if not relevant):\n"
                            + lines
                        )

                    simple_replies_note = (
                        f"\n{t('simple_replies_instruction')}"
                        if simple_replies_enabled
                        else ""
                    )

                    config = types.GenerateContentConfig(
                        system_instruction=filled_system_instruction
                        + f"\n{t('territory_note', territory=territory)}"
                        + mode_note
                        + simple_replies_note
                        + f"\n{language_instruction}"
                        + rag_context,
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
                    if voice_enabled:
                        audio_bytes = synthesize_speech(
                            bot_reply, LANG_TO_TTS_CODE.get(language, "en"))
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")

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
                        st.error(
                            t("quota_error", ref_code=st.session_state.ref_code))
                    else:
                        retry_hint = ""
                        if hasattr(e, "error") and isinstance(e.error, dict):
                            for detail in e.error.get("details", []):
                                if isinstance(detail, dict) and detail.get("@type", "").endswith("RetryInfo"):
                                    retry_delay = detail.get("retryDelay")
                                    if retry_delay:
                                        retry_hint = t(
                                            "retry_hint", retry_delay=retry_delay)
                        st.error(
                            t("gemini_error", error=error_message, retry_hint=retry_hint))
