import datetime
import logging
import os
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
from fastrtc import AdditionalOutputs, audio_to_bytes
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available

from utils.device import get_device, get_torch_and_np_dtypes

# Configure logging
logger = logging.getLogger(__name__)


class Transcriber:
    """
    A class for transcribing audio using ASR models with buffer support.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        force_cpu: bool = False,
        use_bfloat16: bool = False,
    ):
        """
        Initialize the Transcriber.

        Args:
            model_id: The Whisper model ID to use for transcription
            force_cpu: If True, forces CPU usage even if GPU is available
            use_bfloat16: If True, uses bfloat16 precision when available
        """
        self.model_id = model_id
        # Initialize transcript buffer
        self.transcript_buffer = ""
        self.audio_buffer = []
        self.sample_rate = None
        # Setup device and data types
        self.device = get_device(force_cpu=force_cpu)
        self.torch_dtype, self.np_dtype = get_torch_and_np_dtypes(
            self.device, use_bfloat16=use_bfloat16
        )
        logger.info(
            f"Using device: {self.device}, torch_dtype: {self.torch_dtype}, "
            f"np_dtype: {self.np_dtype}"
        )

        # Initialize model, processor and pipeline
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the Whisper model, processor, and pipeline."""
        # Determine attention mechanism
        self.attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        logger.info(f"Using attention: {self.attention}")

        # Load the model
        logger.info(f"Loading Whisper model: {self.model_id}")
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=self.attention,
            )
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading ASR model: {e}")
            logger.error(f"Are you providing a valid model ID? {self.model_id}")
            raise

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Create the pipeline
        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        # Warm up the model
        self._warmup_model()

    def _warmup_model(self) -> None:
        """Warm up the model with empty audio to minimize initial inference latency."""
        logger.info("Warming up Whisper model with dummy input")
        warmup_audio = np.zeros((16000,), dtype=self.np_dtype)  # 1s of silence
        self.pipeline(warmup_audio)
        logger.info("Model warmup complete")

    def append_to_buffer(self, text: str) -> None:
        """
        Append new text to the transcript buffer.

        Args:
            text: The text to append to the buffer
        """
        if self.transcript_buffer and not self.transcript_buffer.endswith(" "):
            self.transcript_buffer += " "
        self.transcript_buffer += text.strip()

    def clear_buffer(self) -> None:
        """Clear the transcript buffer."""
        self.transcript_buffer = ""

    def get_transcript(self) -> str:
        """
        Get the current complete transcript.

        Returns:
            The current transcript buffer
        """
        return self.transcript_buffer

    def clear_buffers(self) -> None:
        """Clear the transcript and audio buffers."""
        self.transcript_buffer = ""
        self.audio_buffer = []

    def append_to_audio_buffer(self, sample_rate: int, audio_array: np.ndarray) -> None:
        """
        Append a new audio chunk to the audio buffer.

        Args:
            sample_rate: Sample rate of the audio
            audio_array: Audio data array
        """
        # Store sample rate if not set
        if self.sample_rate is None:
            self.sample_rate = sample_rate

        # Ensure the sample rates match
        if sample_rate != self.sample_rate:
            logger.warning(
                f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}"
            )
            # We would need to resample here, but for simplicity we'll assume they match

        # Append the audio array to the buffer
        self.audio_buffer.append(audio_array)

    def get_full_audio(self) -> Tuple[int, np.ndarray]:
        """
        Get the complete audio data from buffer.

        Returns:
            Tuple of (sample_rate, concatenated_audio_array)
        """
        if not self.audio_buffer:
            return self.sample_rate, np.array([], dtype=self.np_dtype)

        # Concatenate all audio chunks
        full_audio = np.concatenate(self.audio_buffer, axis=-1)
        return self.sample_rate, full_audio

    def save_session(self, output_dir="saved_sessions"):
        """
        Save the current audio and transcript to files.

        Args:
            output_dir: Directory where files will be saved

        Returns:
            Tuple of (transcript_path, audio_path) or None if nothing to save
        """
        if not self.audio_buffer or not self.transcript_buffer:
            logger.warning("No audio or transcript to save")
            return None

        # Create timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save transcript
        transcript_path = os.path.join(output_dir, f"transcript_{timestamp}.txt")
        with open(transcript_path, "w") as f:
            f.write(self.transcript_buffer)

        # Save audio from the full buffer
        audio_path = os.path.join(output_dir, f"audio_{timestamp}.wav")
        sample_rate, full_audio = self.get_full_audio()

        full_audio = full_audio.flatten()

        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes((full_audio).astype(np.int16).tobytes())

        logger.info(f"Saved transcript to {transcript_path}")
        logger.info(f"Saved audio to {audio_path}")

        return transcript_path, audio_path

    async def transcribe(
        self,
        audio: Tuple[int, np.ndarray],
    ):
        """
        Transcribe audio using the Whisper model and append to buffer.

        Args:
            audio: A tuple containing (sample_rate, audio_array)

        Yields:
            AdditionalOutputs containing the full transcript (buffer + new text)
        """
        sample_rate, audio_array = audio
        self.append_to_audio_buffer(sample_rate, audio_array)

        outputs = self.pipeline(
            audio_to_bytes(audio),
            chunk_length_s=3,
            batch_size=1,
            generate_kwargs={
                "task": "transcribe",
                "language": "english",
            },
        )

        new_text = outputs["text"].strip()

        # Update buffer with new text
        self.append_to_buffer(new_text)

        # Return the complete transcript
        yield AdditionalOutputs(self.get_transcript())

    def unload_model(self):
        """Release model resources to free memory"""
        if hasattr(self, "model") and self.model is not None:
            # Release any model-specific resources
            self.model = None

        # Force garbage collection to free memory
        import gc

        gc.collect()

        return True
