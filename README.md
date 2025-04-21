# Digital Medical Scribe 🎙️📝

A real-time medical transcription and note generation application that combines speech recognition with AI-driven clinical documentation. This project uses FastRTC for live audio streaming, Whisper models for speech recognition, and Ollama LLMs for generating structured medical SOAP notes.

## Features

- **Real-time Speech Transcription** - Instant conversion of doctor-patient conversations to text
- **AI-Powered Clinical Notes** - Generate structured SOAP notes from transcripts
- **Local Model Processing** - All processing happens on your device for privacy
- **Customizable Templates** - Edit and save SOAP note templates
- **Session Management** - Save transcripts and audio recordings for later reference
- **Multiple Models Support** - Switch between different transcription and LLM models

## Demo

## System Requirements

- Python >= 3.10
- ffmpeg (for audio processing)
- Ollama - for local LLM processing
- Sufficient RAM for model loading (8GB minimum recommended)
- GPU recommended for faster transcription (but not required)

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/kunwarsaaim/LocalMediScribe.git
cd LocalMediScribe
```

### Step 2: Install Ollama

Ollama needs to be installed and running on your device for the LLM capabilities.

#### macOS

Download and install from [ollama.com/download](https://ollama.com/download)
(Requires macOS 11 Big Sur or later)

#### Linux

Follow the installation instructions at [ollama.com/download](https://ollama.com/download) for your Linux distribution.

#### Windows

Download and install from [ollama.com/download](https://ollama.com/download)

After installation, Ollama will run as a service in the background.

### Step 3: Install dependencies

#### CPU-only Installation

If you don't have a compatible GPU or prefer to run on CPU only:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate

# Install the project requirements
pip install -r requirements.txt
```

#### GPU Installation

For faster performance with a compatible NVIDIA GPU:

1. **Verify your GPU is detected**:

   ```bash
   nvidia-smi
   ```

   This should display information about your GPU if properly installed.

2. **Install PyTorch with CUDA support**:
   Choose the appropriate CUDA version that matches your system. For CUDA 12.1:

   ```bash
   # Create and activate a virtual environment (recommended)
   python -m venv .env
   source .env/bin/activate  # On Windows: .env\Scripts\activate

   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Install the project requirements
   pip install -r requirements.txt
   ```

   For CUDA 11.8, use:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify the installation**:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
   ```
   This should output `CUDA available: True` if properly configured.

### Step 4: Install ffmpeg

#### macOS

```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

#### Windows

Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install with Chocolatey:

```
choco install ffmpeg
```

### Step 5: Launch the application

```bash
python main.py
```

Once running, navigate to https://localhost:7860 in your browser to start using the application.

## Models

### Speech Transcription Models

The application supports the following Whisper models:

- openai/whisper-large-v3-turbo (default)
- openai/whisper-medium
- openai/whisper-small
- openai/whisper-base

### LLM Models (via Ollama)

For clinical note generation, the following models are supported:

- gemma3:4b (default)
- qwen2.5:7b
- mistral
- phi4-mini

## Usage Guide

1. **Start a session**: Click the microphone button to begin recording the doctor-patient conversation
2. **Review transcript**: The transcript appears in real-time in the left panel
3. **Generate note**: Click "Generate Note" to create a SOAP note from the transcript
4. **Edit note**: Use the "Edit Note" button to make adjustments to the generated note
5. **Save session**: Click "Finish & Save Session" to save the transcript and audio
6. **Save note**: Use "Save Note to File" to export the note as a text file
7. **Reset session**: Clear all data and start a new session with "Reset Session"

## SOAP Note Template

The application uses a customizable SOAP (Subjective, Objective, Assessment, Plan) template for note generation:

```
## Patient Info
**Name:** [Name] | **DOB:** [DOB] | **Date:** [Auto-filled]

## S: Subjective
- **CC:** [Chief complaint]
- **HPI:** [Brief history of present illness]
- **Current Meds:** [Medications]
- **Allergies:** [Allergies]
- **PMH:** [Relevant past medical history]
- **ROS:** [Pertinent positive/negative findings]

## O: Objective
- **Vitals:** T [temp] | BP [BP] | HR [HR] | RR [RR] | SpO2 [O2] | Pain [0-10]
- **Physical Exam:** [Key findings by system]
- **Labs/Studies:** [Relevant results]

## A: Assessment
1. [Primary diagnosis/problem]
2. [Secondary diagnosis/problem]

## P: Plan
- **Diagnostics:** [Ordered tests]
- **Treatment:** [Medications, therapies]
- **Patient Education:** [Instructions given]
- **Follow-up:** [When and circumstances]
```

You can edit this template in the UI to match your specific documentation requirements.

## Project Structure

- `main.py`: Application entry point and UI definition
- `services/`: Core service modules
  - `transcriber.py`: Handles speech recognition using Whisper models
  - `ollama_service.py`: Manages LLM interactions for note generation
- `utils/`: Utility modules
  - `templates.py`: SOAP note templates and system prompts
  - `device.py`: Hardware configuration utilities
- `saved_sessions/`: Storage for saved transcripts, recordings, and notes

## Dependencies

- **fastrtc** - Real-time audio streaming with WebRTC
- **transformers** - Whisper models for speech recognition
- **gradio** - Web interface framework
- **ollama** - Local LLM integration
- **accelerate** - Performance optimization for transformer models

## Privacy Note

All processing happens locally on your device - no data is sent to external servers, making this suitable for handling sensitive medical information in compliance with privacy regulations.

## Limitations

- Performance depends on your hardware (GPU recommended for faster processing)
- Only English language is currently supported for transcription
- Model quality affects the accuracy of both transcription and note generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgements

- [FastRTC](https://github.com/gradio-app/fastrtc) for the real-time audio communication
- [Whisper](https://github.com/openai/whisper) for the speech recognition models
- [Ollama](https://ollama.com/) for local LLM integration
- [Gradio](https://www.gradio.app/) for the web interface
