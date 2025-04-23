# MedicalGPT: Medical Text Simplification App

MedicalGPT converts complex medical terminology into plain language, making healthcare information more accessible to patients. This application utilizes a fine-tuned Mistral-NeMo 12B model with LoRA adaptation to simplify medical texts, translate content into multiple languages, and provide text-to-speech capabilities.

## Features

- **Medical Text Simplification**: Transforms complex medical jargon into easy-to-understand language
- **Multi-language Support**: Translates simplified text into 12 languages
- **Text-to-Speech**: Converts simplified text into spoken audio
- **User-friendly Interface**: Clean, intuitive design for healthcare settings
- **Customizable Settings**: Configure default language and text-to-speech preferences

## Local Installation and Deployment

### Prerequisites

- Python 3.8 or higher
- GPU with at least 16GB VRAM (recommended for optimal performance)
- At least 30GB of free disk space
- Internet connection

### Setup and Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medicalgpt-app.git
   cd medicalgpt-app
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

### Usage Instructions

1. **Loading the Model**:
   - When you first start the application, click the "Load Model" button in the sidebar
   - This will download and initialize the Mistral-NeMo 12B model (may take a few minutes)

2. **Simplifying Medical Text**:
   - Enter complex medical text in the input box
   - Click "Simplify Text" to process the input
   - The simplified output will appear below

3. **Translating Text**:
   - Set a default translation language in the sidebar (optional)
   - Use the language selector and "Translate" button under the output
   - Toggle "Translate output by default" to automatically translate results

4. **Text-to-Speech**:
   - Click the "Listen" button to hear the simplified or translated text
   - Toggle "Enable text-to-speech by default" to automatically play audio

5. **Saving Results**:
   - Use the "Download as Text" button to save the results to a file

## Troubleshooting

- **Out of Memory Errors**: If you encounter memory issues, try:
  - Reducing the length of input text
  - Using a machine with more GPU memory
  - Setting `device_map="cpu"` in the code (will be slower but use less VRAM)

- **Model Loading Issues**:
  - Ensure you have a stable internet connection for the initial model download
  - Check that you have sufficient disk space (model requires ~23GB)

- **Translation or TTS Problems**:
  - Both features require an internet connection
  - Some language combinations may have reduced quality

## System Requirements

- **Minimum**: 16GB RAM, CPU only (very slow inference)
- **Recommended**: 32GB RAM, GPU with 16GB+ VRAM
- **Storage**: At least 30GB free disk space

## Model Information

- Base Model: Mistral-NeMo 12B
- Fine-tuning: Low-Rank Adaptation (LoRA) on 800 pairs of medical notes
- Performance: 92% SARI score on test data
- Inference: Runs on CPU or GPU with reduced precision (bfloat16)

## Credits

This application was developed as part of the Medical Text Simplification Project, aiming to improve healthcare communication accessibility. The model was fine-tuned on a dataset of medical notes from five clinical domains: Gastroenterology, Neurology, Orthopedics, Radiology, and Urology.