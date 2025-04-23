import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import platform
import sys
from gtts import gTTS
import base64
import io
from googletrans import Translator
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="MedicalGPT: Text Simplifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        text-align: center;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .output-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-top: 20px;
    }
    .input-container {
        background-color: #f5f7f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e5ea;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .feature-icon {
        font-size: 24px;
        margin-right: 10px;
        color: #3498db;
    }
    #audio-output {
        margin-top: 10px;
    }
    .footer {
        margin-top: 40px;
        text-align: center;
        color: #95a5a6;
        font-size: 0.9rem;
    }
    .debug-info {
        font-family: monospace;
        font-size: 0.85em;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'tts_enabled' not in st.session_state:
    st.session_state.tts_enabled = True
if 'default_language' not in st.session_state:
    st.session_state.default_language = "English"
if 'simplified_text' not in st.session_state:
    st.session_state.simplified_text = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'current_language' not in st.session_state:
    st.session_state.current_language = "English"
if 'translation_enabled' not in st.session_state:
    st.session_state.translation_enabled = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'api_mode' not in st.session_state:
    st.session_state.api_mode = "local"
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = ""
if 'api_key_saved' not in st.session_state:
    st.session_state.api_key_saved = False
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Header
st.markdown("<h1 class='main-header'>MedicalGPT</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Converting complex medical terminology into plain language</p>", unsafe_allow_html=True)

# Function to save HF token to session state
def save_token():
    if st.session_state.hf_token:
        st.session_state.api_key_saved = True
        st.success("API key saved for this session")

# Function for text-to-speech
def text_to_speech(text, language_code="en"):
    # Language code mapping
    language_map = {
        "English": "en", "Spanish": "es", "French": "fr", 
        "German": "de", "Chinese": "zh-CN", "Japanese": "ja",
        "Russian": "ru", "Arabic": "ar", "Portuguese": "pt",
        "Hindi": "hi", "Bengali": "bn", "Korean": "ko"
    }
    
    # Get the correct language code
    lang_code = language_map.get(st.session_state.current_language, "en")
    
    # Use the appropriate text based on current language
    if st.session_state.current_language == "English":
        tts_text = st.session_state.simplified_text
    else:
        tts_text = st.session_state.translated_text or st.session_state.simplified_text
        
    try:
        tts = gTTS(text=tts_text, lang=lang_code, slow=False)
        
        # Save to BytesIO object
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Encode to base64 for HTML audio element
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        
        # Create HTML audio element
        audio_html = f'<audio id="audio-output" controls autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        return audio_html
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

# Function to translate text
def translate_text(text, target_language):
    if target_language == "English":
        return text
        
    # Language code mapping
    language_map = {
        "Spanish": "es", "French": "fr", "German": "de", 
        "Chinese": "zh-cn", "Japanese": "ja", "Russian": "ru",
        "Arabic": "ar", "Portuguese": "pt", "Hindi": "hi",
        "Bengali": "bn", "Korean": "ko"
    }
    
    try:
        translator = Translator()
        translation = translator.translate(text, dest=language_map.get(target_language, "en"))
        return translation.text
    except Exception as e:
        st.error(f"Translation Error: {str(e)}")
        return text

# Function to get system info for debugging
def get_system_info():
    info = {
        "Platform": platform.platform(),
        "Python Version": sys.version,
        "Torch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "GPU Devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["GPU Name"] = torch.cuda.get_device_name(0)
        info["GPU Memory"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    
    return info

# Function to load the model
def load_model():
    try:
        with st.spinner("Loading model... This may take a minute."):
            # Model configuration
            model_name = "shikhac30/mistral_nemo_12b_medical_model"
            
            # Load using alternative method
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # More conservative loading settings
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Changed from bfloat16 to float16 for better compatibility
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                revision="main"  # Explicitly use main branch
            )
            
            # Build the pipeline
            pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer
            )
            
            # Store in session state
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.pipe = pipe
            st.session_state.model_loaded = True
            
            return True
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error loading model: {error_msg}")
        
        if st.session_state.debug_mode:
            st.markdown("<div class='debug-info'>", unsafe_allow_html=True)
            st.write("### Debug Information")
            st.write(get_system_info())
            st.write("### Error Details")
            st.write(error_msg)
            st.markdown("</div>", unsafe_allow_html=True)
            
        return False

# Function to call Hugging Face Inference API
def call_hf_api(input_text):
    try:
        if not st.session_state.hf_token:
            st.error("Please enter your HuggingFace API token in the sidebar")
            return None
            
        with st.spinner("Calling Hugging Face Inference API..."):
            # Format input
            formatted_input = f"<s>[INST] {input_text.strip()} [/INST]"
            
            # API endpoint
            API_URL = f"https://api-inference.huggingface.co/models/shikhac30/mistral_nemo_12b_medical_model"
            
            # Make the request
            headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
            payload = {"inputs": formatted_input, "parameters": {"max_new_tokens": 128, "temperature": 0.7, "do_sample": True}}
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code != 200:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
            # Parse the response
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                output_text = generated_text.replace(formatted_input, "").strip()
                return output_text
            else:
                st.error(f"Unexpected API response format: {result}")
                return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# Function to simplify text using the model
def simplify_medical_text(input_text):
    # Use API mode if selected
    if st.session_state.api_mode == "api":
        return call_hf_api(input_text)
    
    try:
        # Ensure model is loaded
        if not st.session_state.model_loaded:
            success = load_model()
            if not success:
                return None
        
        with st.spinner("Simplifying medical text..."):
            # Format input text
            formatted_input = f"<s>[INST] {input_text.strip()} [/INST]"
            
            # Generate response
            result = st.session_state.pipe(
                formatted_input, 
                max_new_tokens=128, 
                do_sample=True, 
                temperature=0.7
            )
            
            # Extract generated text
            output_text = result[0]["generated_text"].replace(formatted_input, "").strip()
            
            return output_text
            
    except Exception as e:
        st.error(f"Error during text simplification: {str(e)}")
        if st.session_state.debug_mode:
            st.markdown("<div class='debug-info'>", unsafe_allow_html=True)
            st.write("### Debug Information")
            st.write(get_system_info())
            st.write("### Error Details")
            st.write(str(e))
            st.markdown("</div>", unsafe_allow_html=True)
        return None

# Function to handle the simplify button click
def on_simplify_click():
    if not st.session_state.medical_text.strip():
        st.warning("Please enter some medical text to simplify")
        return
        
    # Call the simplification function
    simplified = simplify_medical_text(st.session_state.medical_text)
    
    if simplified:
        st.session_state.simplified_text = simplified
        st.session_state.current_language = "English"
        st.session_state.translated_text = ""
        
        # If translation is enabled by default, translate immediately
        if st.session_state.translation_enabled:
            with st.spinner(f"Translating to {st.session_state.default_language}..."):
                st.session_state.translated_text = translate_text(simplified, st.session_state.default_language)
                st.session_state.current_language = st.session_state.default_language

# Function to handle language change
def on_language_change():
    if not st.session_state.simplified_text:
        return
        
    if st.session_state.language_selector != st.session_state.current_language:
        if st.session_state.language_selector == "English":
            st.session_state.translated_text = ""
        else:
            with st.spinner(f"Translating to {st.session_state.language_selector}..."):
                st.session_state.translated_text = translate_text(
                    st.session_state.simplified_text, 
                    st.session_state.language_selector
                )
        st.session_state.current_language = st.session_state.language_selector

# Function to toggle API mode
def toggle_api_mode():
    st.session_state.model_loaded = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model loading options
    st.subheader("Model Options")
    
    # Option to use HF API instead of local model
    api_mode = st.radio(
        "How to run the model:",
        options=["Local (GPU/CPU)", "Hugging Face API"],
        index=0 if st.session_state.api_mode == "local" else 1,
        help="Local mode loads the model on your machine. API mode uses Hugging Face's servers."
    )
    
    st.session_state.api_mode = "local" if api_mode == "Local (GPU/CPU)" else "api"
    
    # Show API token input if API mode is selected
    if st.session_state.api_mode == "api":
        if not st.session_state.api_key_saved:
            st.session_state.hf_token = st.text_input(
                "HuggingFace API Token", 
                type="password",
                help="Enter your HuggingFace API token to access the model"
            )
            st.button("Save API Key", on_click=save_token)
        else:
            st.success("API key is saved for this session")
            if st.button("Change API Key"):
                st.session_state.api_key_saved = False
    
    # Local model loading button
    if st.session_state.api_mode == "local":
        if not st.session_state.model_loaded:
            st.warning("Model not loaded yet. You'll need to load it before simplifying text.")
            if st.button("Load Model"):
                load_model()
        else:
            st.success("Model loaded successfully!")
    
    st.divider()
    
    # Default settings
    st.subheader("Default Settings")
    
    # TTS option
    st.session_state.tts_enabled = st.toggle(
        "Enable text-to-speech by default",
        value=st.session_state.tts_enabled
    )
    
    # Translation options
    st.session_state.translation_enabled = st.toggle(
        "Translate output by default",
        value=st.session_state.translation_enabled
    )
    
    # Only show language selector if translation is enabled
    if st.session_state.translation_enabled:
        st.session_state.default_language = st.selectbox(
            "Default language",
            options=["English", "Spanish", "French", "German", "Chinese", 
                    "Japanese", "Russian", "Arabic", "Portuguese", 
                    "Hindi", "Bengali", "Korean"],
            index=0
        )
    
    st.divider()
    
    # Debug mode toggle
    st.session_state.debug_mode = st.toggle(
        "Debug mode",
        value=st.session_state.debug_mode
    )
    
    if st.session_state.debug_mode:
        st.markdown("<div class='debug-info'>", unsafe_allow_html=True)
        st.write("### System Information")
        for key, value in get_system_info().items():
            st.write(f"**{key}:** {value}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # About section
    st.subheader("About")
    st.markdown("""
    **MedicalGPT** uses a fine-tuned Mistral-NeMo 12B model that was trained on 800 pairs of medical notes and their simplified versions.
    
    The model uses Low-Rank Adaptation (LoRA) to adapt the base model for medical text simplification, achieving a SARI score of 92%.
    """)

# Main content area
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
st.text_area(
    "Enter medical text to simplify:",
    key="medical_text",
    height=150,
    placeholder="Paste complex medical text here. Example: 'The patient presents with acute myocardial infarction with elevated troponin levels...'"
)
st.markdown("</div>", unsafe_allow_html=True)

# Submit button
submit_button = st.button("Simplify Text", on_click=on_simplify_click, type="primary", use_container_width=True)
if submit_button and st.session_state.api_mode == "local" and not st.session_state.model_loaded:
    st.warning("Please load the model first using the button in the sidebar.")
elif submit_button and st.session_state.api_mode == "api" and not st.session_state.api_key_saved and not st.session_state.hf_token:
    st.warning("Please enter your HuggingFace API token in the sidebar.")

# Output area (only display if there's content)
if st.session_state.simplified_text:
    st.markdown("<div class='output-container'>", unsafe_allow_html=True)
    
    # Output header with language indicator
    language_display = "English"
    if st.session_state.current_language != "English":
        language_display = f"{st.session_state.current_language}"
    
    st.subheader(f"Simplified Text ({language_display}):")
    
    # Display the appropriate text based on current language
    if st.session_state.current_language == "English":
        st.write(st.session_state.simplified_text)
    else:
        st.write(st.session_state.translated_text)
    
    # Action buttons row
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # TTS button
        if st.button("🔊 Listen", use_container_width=True):
            audio_html = text_to_speech(
                st.session_state.translated_text if st.session_state.translated_text else st.session_state.simplified_text,
                st.session_state.current_language
            )
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)
    
    with col2:
        # Language selection and translate button
        cols = st.columns([3, 1])
        with cols[0]:
            st.selectbox(
                "Language",
                options=["English", "Spanish", "French", "German", "Chinese", 
                        "Japanese", "Russian", "Arabic", "Portuguese", 
                        "Hindi", "Bengali", "Korean"],
                index=["English", "Spanish", "French", "German", "Chinese", 
                      "Japanese", "Russian", "Arabic", "Portuguese", 
                      "Hindi", "Bengali", "Korean"].index(st.session_state.current_language),
                key="language_selector"
            )
        with cols[1]:
            st.button("Translate", on_click=on_language_change, use_container_width=True)
    
    # Auto TTS on output generation
    if st.session_state.tts_enabled and 'last_text' not in st.session_state or st.session_state.get('last_text') != (st.session_state.translated_text or st.session_state.simplified_text):
        audio_html = text_to_speech(
            st.session_state.translated_text if st.session_state.translated_text else st.session_state.simplified_text,
            st.session_state.current_language
        )
        if audio_html:
            st.markdown(audio_html, unsafe_allow_html=True)
            st.session_state['last_text'] = st.session_state.translated_text or st.session_state.simplified_text
    
    # Download buttons
    st.download_button(
        label="📄 Download as Text",
        data=st.session_state.translated_text if st.session_state.translated_text else st.session_state.simplified_text,
        file_name=f"simplified_medical_text_{st.session_state.current_language.lower()}.txt",
        mime="text/plain"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown(
    "MedicalGPT: Breaking down barriers in healthcare communication. "
    "© 2025 Medical Text Simplification Project"
)
st.markdown("</div>", unsafe_allow_html=True)