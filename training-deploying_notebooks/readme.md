# Fine-Tuning & Deploying Mistral-Nemo-Instruct-2407 to Simplify Medical Text 

This project fine-tunes a Mistral 12B language model to simplify complex medical text, making it more accessible to patients and non-specialists. The implementation is divided into four notebooks, each handling a distinct stage of the workflow.

## Stage 1: Fine-Tuning (stage1_mistral-12b-finetunedetail.ipynb)

This notebook prepares and executes the fine-tuning process:

- Sets up the environment and imports necessary libraries
- Loads a medical text simplification dataset containing pairs of original and simplified medical texts
- Formats the data with instruction tokens (`<s>[INST]` and `[/INST]`) for fine-tuning
- Loads the Mistral-Nemo-Instruct-2407 base model with BF16 precision
- Configures LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
  - Uses rank 64 for low-rank matrices
  - Targets attention and gate projection layers
- Sets up training hyperparameters (learning rate, epochs, batch size)
- Trains the model using the SFTTrainer (Supervised Fine-Tuning)
- Saves and uploads the model adapter to Hugging Face

## Stage 2: Model Merging (stage2_merge_model.ipynb)

This notebook handles the merging of the adapter and base model:

- Loads both the base model (Mistral-Nemo-Instruct-2407) and the fine-tuned adapter
- Merges the adapter weights into the base model (effectively applying the learned parameters)
- Saves the merged model locally
- Pushes the complete model to Hugging Face for sharing and deployment

## Stage 3: Interactive Testing (stage3_live_chat.ipynb)

This notebook sets up an interface to test the model interactively:

- Loads the merged model and tokenizer
- Creates a text generation pipeline for inference
- Implements a simple chat interface using Rich library for formatted output
- Tests the model with a sample medical text prompt about a patient with pulmonary edema
- Displays both the input prompt and simplified output with visual formatting

## Stage 4: Evaluation (stage4_evalll.ipynb)

This notebook provides comprehensive evaluation of the model's performance:

- Tests the model on multiple medical text samples
- Uses a GPT-based external evaluator to rate simplification quality on a scale of 1-10
- Measures generation time for performance benchmarking
- Displays both original and simplified texts side by side
- Shows that the model consistently achieves high simplification scores (mostly 8/10)

The model demonstrates strong capability in translating complex medical terminology into simpler, patient-friendly language while preserving the essential medical information.