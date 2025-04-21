import datetime
import logging
import os

import gradio as gr
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    SileroVadOptions,
    WebRTC,
)

from services.ollama_service import OllamaService
from services.transcriber import Transcriber
from utils.templates import SYSTEM_PROMPT, get_template_with_date, save_custom_template

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicalScribeApp:
    # Available models
    TRANSCRIPTION_MODELS = [
        "openai/whisper-large-v3-turbo",
        "openai/whisper-medium",
        "openai/whisper-small",
        "openai/whisper-base",
    ]

    OLLAMA_MODELS = ["gemma3:4b", "qwen2.5:7b", "mistral", "phi4-mini"]

    def __init__(self):
        # Default models
        self.current_transcription_model = "openai/whisper-large-v3-turbo"
        self.current_ollama_model = "gemma3:4b"

        self.default_template = get_template_with_date()

        # Initialize services with default models
        self.initialize_services()
        logger.info("Initializing FastRTC stream")

        self.is_generating_note = False

    def initialize_services(self):
        """Initialize or reinitialize services with selected models"""
        try:
            if hasattr(self, "transcriber"):
                logger.info("Releasing previous transcription model...")
                if hasattr(self.transcriber, "unload_model"):
                    self.transcriber.unload_model()
                self.transcriber = None

            if hasattr(self, "ollama_service"):
                logger.info("Releasing previous LLM model...")
                if hasattr(self.ollama_service, "unload_model"):
                    self.ollama_service.unload_model()
                self.ollama_service = None

            import gc

            gc.collect()

            logger.info(
                f"Initializing transcriber with model: {self.current_transcription_model}"
            )
            self.transcriber = Transcriber(model_id=self.current_transcription_model)

            logger.info(f"Loading Ollama model: {self.current_ollama_model}")
            self.ollama_service = OllamaService(model_name=self.current_ollama_model)
            self.ollama_service.load_model()

            return f"Models loaded: {self.current_transcription_model} (transcription), {self.current_ollama_model} (LLM)"
        except MemoryError:
            logger.error("Not enough memory to load the requested models!")
            return (
                "Error: Insufficient memory to load models. Try using smaller models."
            )
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            return f"Error loading models: {str(e)}"

    def save_session_callback(self):
        """Function to handle the Save button click"""
        paths = self.transcriber.save_session()
        if paths:
            transcript_path, audio_path = paths
            return f"Session saved:\nTranscript: {transcript_path}\nAudio: {audio_path}"
        return "Nothing to save"

    def reset_session_callback(self):
        """Function to reset the session"""
        self.transcriber.clear_buffers()
        # Return empty strings to clear transcript and note displays
        return (
            "",
            "",
            gr.update(visible=False),
        )  # Clear transcript, note output, and hide note buttons

    def change_models_callback(self, transcription_model, ollama_model):
        """Function to change models based on dropdown selections"""
        reload_needed = (
            self.current_transcription_model != transcription_model
            or self.current_ollama_model != ollama_model
        )

        if reload_needed:
            self.current_transcription_model = transcription_model
            self.current_ollama_model = ollama_model
            return self.initialize_services()
        else:
            return "No change needed - models already loaded"

    def generate_note_callback(self, transcript_text, template_text):
        """Function to generate a clinical note from the transcript with streaming output"""
        if not transcript_text:
            yield "No transcript available to generate a note."
            return

        system_prompt = SYSTEM_PROMPT

        user_prompt = f"""Here is the conversation transcript:
                        {transcript_text}
                        Please format the clinical note using this template:
                        {template_text}"""

        # Generate the note (streaming)
        note = ""
        try:
            stream = self.ollama_service.generate_note(
                transcript=user_prompt, system_prompt=system_prompt, stream=True
            )

            # Stream each chunk as it arrives
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    note += content
                    yield note
        except Exception as e:
            logger.error(f"Error generating note: {e}")
            yield f"Error generating note: {str(e)}"

    def save_note_callback(self, note_text):
        """Function to save the clinical note to a file"""
        if not note_text:
            return "No note to save"

        # Create timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "saved_sessions"

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save note to file
        note_path = os.path.join(output_dir, f"note_{timestamp}.txt")
        with open(note_path, "w") as f:
            f.write(note_text)

        logger.info(f"Saved clinical note to {note_path}")
        return f"Note saved to: {note_path}"

    def toggle_edit_mode(self, is_editing, note_content):
        """Function to toggle between view and edit modes for notes"""
        if not is_editing:
            return (
                True,
                note_content,
                gr.update(visible=False),
                gr.update(visible=True, value=note_content),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        else:
            return is_editing, note_content, None, None, None, None

    def save_edited_note(self, note_text):
        """Function to save edits made to notes"""
        return (
            False,
            note_text,
            gr.update(visible=True, value=note_text),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def generate_note_with_ui_update(self, transcript_text, template_text):
        """Function to generate a note with UI updates for loading indicators"""
        if not transcript_text:
            yield [
                "No transcript available to generate a note.",
                gr.update(visible=False),
            ]
            return

        gr.update(visible=True)

        system_prompt = SYSTEM_PROMPT

        # Prepare user prompt with the template
        user_prompt = f"""Here is the conversation transcript:
                        {transcript_text}
                        Please format the clinical note using this template:
                        {template_text}"""

        # Generate the note (streaming)
        note = ""
        try:
            self.is_generating_note = True
            yield ["", gr.update(visible=True)]
            stream = self.ollama_service.generate_note(
                transcript=user_prompt, system_prompt=system_prompt, stream=True
            )

            # Stream each chunk as it arrives
            first_chunk = True
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    note += content

                    # Hide loading indicator after first content arrives
                    if first_chunk:
                        first_chunk = False
                        yield [note, gr.update(visible=False)]
                    else:
                        yield [note, gr.update(visible=False)]
        except Exception as e:
            logger.error(f"Error generating note: {e}")
            yield [f"Error generating note: {str(e)}", gr.update(visible=False)]
        finally:
            self.is_generating_note = False

    def save_template_as_default(self, template_text):
        """Function to save the current template as the default template"""
        if not template_text:
            return "Template is empty. Cannot save."

        success = save_custom_template(template_text)
        if success:
            logger.info("Default template updated successfully")
            return "Template saved as default successfully!"
        else:
            logger.error("Failed to save default template")
            return "Error: Failed to save template as default."

    def build_and_launch_ui(self):
        """Build and launch the Gradio UI"""
        server_name = os.getenv("SERVER_NAME", "localhost")
        port = int(os.getenv("PORT", 7860))

        logger.info(f"Launching Gradio UI on {server_name}:{port}")
        with gr.Blocks(theme=gr.themes.Default()) as ui:
            # State variable for note editing mode
            is_editing_note = gr.State(False)
            editable_note_content = gr.State("")

            with gr.Row():
                # Left side (Input area)
                with gr.Column(scale=2):
                    # Model selection dropdowns
                    with gr.Row():
                        transcription_dropdown = gr.Dropdown(
                            choices=self.TRANSCRIPTION_MODELS,
                            value=self.current_transcription_model,
                            label="Transcription Model",
                        )
                        ollama_dropdown = gr.Dropdown(
                            choices=self.OLLAMA_MODELS,
                            value=self.current_ollama_model,
                            label="LLM Model",
                        )

                    change_models_btn = gr.Button("Change Models")

                    audio = WebRTC(
                        label="Stream",
                        mode="send",
                        modality="audio",
                        height=200,
                    )

                    with gr.Row():
                        finish_btn = gr.Button("Finish & Save Session")
                        reset_btn = gr.Button("Reset Session")

                    transcript = gr.Textbox(label="Transcript", interactive=False)

                    # Status elements moved to bottom left
                    with gr.Row():
                        model_status = gr.Textbox(
                            label="Model Status",
                            value=f"Models loaded: {self.current_transcription_model} (transcription), {self.current_ollama_model} (LLM)",
                            interactive=False,
                        )

                    save_status = gr.Textbox(label="Status", interactive=False)

                # Right side (Output area)
                with gr.Column(scale=3):
                    # Generate button
                    generate_note_btn = gr.Button("Generate Note", size="lg")

                    # Loading indicator - using HTML for a spinner
                    loading_indicator = gr.HTML(
                        """<div style="display: flex; justify-content: center; margin: 10px;">
                           <div class="loading-spinner">
                             <style>
                               .loading-spinner {
                                 border: 5px solid rgba(0, 0, 0, 0.1);
                                 width=36px;
                                 height=36px;
                                 border-radius=50%;
                                 border-left-color=#09f;
                                 animation=spin 1s ease infinite;
                               }
                               @keyframes spin {
                                 0% { transform: rotate(0deg); }
                                 100% { transform: rotate(360deg); }
                               }
                             </style>
                           </div>
                           <div style="margin-left: 10px; font-weight: bold;">Generating note...</div>
                         </div>""",
                        visible=False,
                    )

                    with gr.Group():
                        note_output = gr.Markdown(label="Generated Note")
                        note_text_edit = gr.TextArea(
                            label="Edit Note", visible=False, lines=20
                        )

                        with gr.Row(visible=False) as note_buttons_row:
                            edit_note_btn = gr.Button("Edit Note")
                            save_edits_btn = gr.Button("Save Edits", visible=False)
                            save_note_btn = gr.Button("Save Note to File")

                    template = gr.Textbox(
                        label="SOAP Note Template",
                        interactive=True,
                        value=self.default_template,
                        lines=15,
                    )

                    # Add button to save template as default
                    save_template_btn = gr.Button("Save as Default Template")
                    save_template_btn.click(
                        fn=self.save_template_as_default,
                        inputs=[template],
                        outputs=[save_status],
                    )

            audio.stream(
                ReplyOnPause(
                    self.transcriber.transcribe,
                    algo_options=AlgoOptions(
                        audio_chunk_duration=0.6,
                        started_talking_threshold=0.2,
                        speech_threshold=0.1,
                    ),
                    model_options=SileroVadOptions(
                        threshold=0.5,
                        min_speech_duration_ms=250,
                        max_speech_duration_s=30,
                        min_silence_duration_ms=2000,
                        window_size_samples=1024,
                        speech_pad_ms=400,
                    ),
                ),
                inputs=[audio],
                outputs=[audio],
            )

            audio.on_additional_outputs(
                lambda current: current,
                outputs=[transcript],
            )

            change_models_btn.click(
                fn=self.change_models_callback,
                inputs=[transcription_dropdown, ollama_dropdown],
                outputs=[model_status],
            )
            finish_btn.click(fn=self.save_session_callback, outputs=[save_status])

            # Update the reset button to clear transcript, note output, and hide buttons
            reset_btn.click(
                fn=self.reset_session_callback,
                outputs=[transcript, note_output, note_buttons_row],
            )

            # Generate note button with output and UI update
            generate_note_btn.click(
                fn=self.generate_note_with_ui_update,
                inputs=[transcript, template],
                outputs=[note_output, loading_indicator],
                api_name=False,
                queue=True,
            ).then(
                # After note generation, show the edit buttons
                lambda: gr.update(visible=True),
                None,
                [note_buttons_row],
            )

            # Edit and save note functionality
            edit_note_btn.click(
                fn=self.toggle_edit_mode,
                inputs=[is_editing_note, note_output],
                outputs=[
                    is_editing_note,
                    editable_note_content,
                    note_output,
                    note_text_edit,
                    edit_note_btn,
                    save_edits_btn,
                ],
            )

            save_edits_btn.click(
                fn=self.save_edited_note,
                inputs=[note_text_edit],
                outputs=[
                    is_editing_note,
                    editable_note_content,
                    note_output,
                    note_text_edit,
                    edit_note_btn,
                    save_edits_btn,
                ],
            )

            save_note_btn.click(
                fn=self.save_note_callback,
                inputs=[note_output],
                outputs=[save_status],
            )

            # Launch the UI
            ui.launch(
                server_port=port, server_name=server_name, ssl_verify=False, debug=True
            )


if __name__ == "__main__":
    app = MedicalScribeApp()
    app.build_and_launch_ui()
