import streamlit as st
import requests
from gtts import gTTS # Google Text-to-Speech
from googletrans import Translator, LANGUAGES # Google Translate
import io # To handle audio bytes in memory

# --- Configuration ---
API_URL = "http://17.26.71.21:123/v1/completions" # Replace to your own API or load using HF like previous version
MODEL_NAME = "shikhac30/mistral_nemo_12b_medical_model"
DEFAULT_LANGUAGES = {"en": "english", "es": "spanish", "fr": "french", "de": "german", "hi": "hindi", "ar": "arabic", "pt": "portuguese", "zh-cn": "chinese (simplified)"}

# --- Helper Functions (No changes needed from previous version) ---
def simplify_text_api(prompt_text: str) -> str | None:
    """Calls the deployed LLM API to simplify the text."""
    formatted_input = f"<s>[INST] {prompt_text.strip()} [/INST]"
    headers = {'Content-Type': 'application/json'}
    json_data = {
        'model': MODEL_NAME,
        'prompt': formatted_input,
        'max_tokens': 150,
        'temperature': 0.7
    }
    try:
        response = requests.post(API_URL, headers=headers, json=json_data, timeout=60)
        response.raise_for_status()
        generated_text = response.json()["choices"][0]["text"].strip()
        return generated_text
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Failed to connect or get response. Details: {e}")
        return None
    except KeyError:
        st.error("API Error: Unexpected response format.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during simplification: {e}")
        return None

def text_to_speech(text: str, lang: str = 'en') -> io.BytesIO | None:
    """Generates TTS audio using gTTS and returns audio bytes."""
    gtts_lang_code = lang
    if lang == 'zh-cn': gtts_lang_code = 'zh-CN'
    try:
        tts = gTTS(text=text, lang=gtts_lang_code, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"TTS Error: Could not generate audio for lang '{gtts_lang_code}'. Details: {e}")
        return None

def translate_text(text: str, target_lang: str = 'en') -> str | None:
    """Translates text using googletrans."""
    translator_lang_code = target_lang
    try:
        translator = Translator()
        translated = translator.translate(text, dest=translator_lang_code)
        return translated.text
    except Exception as e:
        st.error(f"Translation Error: Could not translate to lang '{translator_lang_code}'. Details: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="MedicalGPT: Text Simplifier",
    page_icon="🩺",
    layout="wide"
)

# --- Session State Initialization ---
# Settings
if 'autoplay_tts' not in st.session_state:
    st.session_state.autoplay_tts = False
if 'default_translate' not in st.session_state:
    st.session_state.default_translate = False
if 'default_translate_lang' not in st.session_state:
    st.session_state.default_translate_lang = 'es'
# Chat and Last State
if "messages" not in st.session_state:
    # Message structure: {'role': str, 'content': str, 'audio_en': BytesIO|None, 'audio_tr': BytesIO|None, 'lang_tr': str|None}
    st.session_state.messages = []
if "last_simplified_english" not in st.session_state:
    st.session_state.last_simplified_english = None # Stores the original simplified English text

# --- Sidebar ---
st.sidebar.header("⚙️ Settings & Actions")
st.sidebar.markdown("Configure default behavior and trigger actions.")

# Renamed Checkbox for Autoplay (Item 3)
st.session_state.autoplay_tts = st.sidebar.checkbox(
    "🔊 Autoplay generated audio",
    value=st.session_state.autoplay_tts,
    help="If checked, audio will attempt to play automatically. Behavior depends on browser permissions. If default translation is on, translated audio is prioritized for autoplay."
)
st.session_state.default_translate = st.sidebar.checkbox(
    "🌐 Translate by default",
    value=st.session_state.default_translate
)

# Language selection always visible for default OR manual translation trigger
st.session_state.default_translate_lang = st.sidebar.selectbox(
    "Translation Language",
    options=list(DEFAULT_LANGUAGES.keys()),
    format_func=lambda x: f"{DEFAULT_LANGUAGES.get(x, x).capitalize()} ({x})",
    index=list(DEFAULT_LANGUAGES.keys()).index(st.session_state.default_translate_lang)
)

# Manual Translation Trigger Button (Item 2)
if st.sidebar.button("🌐 Translate", help="Translates the last simplified English text using the language selected above."):
    if st.session_state.last_simplified_english:
        target_lang = st.session_state.default_translate_lang
        lang_name = DEFAULT_LANGUAGES.get(target_lang, target_lang).capitalize()
        if target_lang == 'en':
             st.sidebar.warning("Select a language other than English to translate.")
        else:
            with st.spinner(f"Translating last message to {lang_name}..."):
                manual_translated_text = translate_text(st.session_state.last_simplified_english, target_lang=target_lang)
                if manual_translated_text:
                    # Generate TTS for this manual translation
                    audio_bytes_manual_tr = text_to_speech(manual_translated_text, lang=target_lang)

                    # Construct the message content
                    manual_message_content = f"**Manual Translation ({lang_name}):**\n\n{manual_translated_text}"

                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": manual_message_content,
                        "audio_en": None, # No English audio for this specific message
                        "audio_tr": audio_bytes_manual_tr,
                        "lang_tr": target_lang
                    })
                    # Rerun to display the new message
                    st.rerun()
                else:
                    st.sidebar.error(f"Failed to translate to {lang_name}.")
    else:
        st.sidebar.warning("Simplify some text first before translating.")

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About MedicalGPT")
st.sidebar.markdown("""
**MedicalGPT** simplifies complex medical text using a fine-tuned Mistral-NeMo 12B model.

This model was trained on approximately 800 pairs of medical notes and their corresponding simplified versions.

Low-Rank Adaptation (LoRA) was employed to efficiently adapt the base LLM for this specific task, achieving strong performance in making medical language more accessible.
""")


# --- Main App Area ---
st.title("🩺 MedicalGPT")
st.markdown("##### Converting complex medical terminology into plain language.")
st.markdown("---")

# Display chat history
# Use enumerate to get the index (i) for unique keys
for message in (st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Determine autoplay based on settings and message content
        autoplay_en = (st.session_state.autoplay_tts and
                       message.get("audio_en") is not None and
                       message.get("audio_tr") is None)

        autoplay_tr = (st.session_state.autoplay_tts and
                       message.get("audio_tr") is not None)

        if message.get("audio_en"):
            st.markdown("**Listen (English):**")
            # Add unique key using message index 'i'
            st.audio(message["audio_en"], format='audio/mp3', autoplay=autoplay_en)

        if message.get("audio_tr"):
            lang_tr_code = message.get("lang_tr", "translation")
            lang_tr_name = DEFAULT_LANGUAGES.get(lang_tr_code, lang_tr_code).capitalize()
            st.markdown(f"**Listen ({lang_tr_name}):**")
            st.audio(message["audio_tr"], format='audio/mp3', autoplay=autoplay_tr)

# Input field for user prompt
prompt = st.chat_input("Enter complex medical text here...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "audio_en": None, "audio_tr": None, "lang_tr": None})
    with st.chat_message("user"):
        st.write(prompt)

    # Simplify Text
    with st.spinner("🔄 Simplifying text..."):
        original_simplified_text = simplify_text_api(prompt) 

    # Process response
    if original_simplified_text:
        st.session_state.last_simplified_english = original_simplified_text
        assistant_display_content = f"**Simplified Text:**\n\n{original_simplified_text}"
        audio_bytes_en = None
        audio_bytes_tr = None
        lang_tr = None

        # Always generate English TTS (Item 3)
        with st.spinner("🔊 Generating English audio..."):
            audio_bytes_en = text_to_speech(original_simplified_text, lang='en')

        # Handle Default Translation
        if st.session_state.default_translate:
            target_lang = st.session_state.default_translate_lang
            lang_name = DEFAULT_LANGUAGES.get(target_lang, target_lang).capitalize()
            if target_lang != 'en': # Avoid translating English to English
                with st.spinner(f"🌐 Translating to {lang_name}..."):
                    default_translated_text = translate_text(original_simplified_text, target_lang=target_lang)

                if default_translated_text:
                    assistant_display_content += f"\n\n---\n**Simplified Text (Translated - {lang_name}):**\n\n{default_translated_text}"
                    lang_tr = target_lang
                    # Always generate Translated TTS if default translation is on (Item 3)
                    with st.spinner(f"🔊 Generating {lang_name} audio..."):
                        audio_bytes_tr = text_to_speech(default_translated_text, lang=target_lang)
                else:
                    assistant_display_content += f"\n\n---\n*Translation to {lang_name} failed.*"
            else:
                 assistant_display_content += f"\n\n---\n*(Default translation set to English)*"


        # Add assistant message with all generated content and audio
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_display_content,
            "audio_en": audio_bytes_en,
            "audio_tr": audio_bytes_tr,
            "lang_tr": lang_tr
        })
        st.rerun() # Rerun to display the new message including audio players

    else:
        st.error("Failed to get simplified text. Please check input or API status.")