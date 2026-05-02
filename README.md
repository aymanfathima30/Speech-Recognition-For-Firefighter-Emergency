"""
Speech Recognition System for Firefighters
High-noise emergency environment speech processing with NLP urgency detection.
"""

import numpy as np
import librosa
import soundfile as sf
from pipeline.preprocessor import AudioPreprocessor
from pipeline.cnn_model import CNNSpeechModel
from pipeline.urgency_detector import UrgencyDetector
from pipeline.dispatcher import FirefighterDispatcher
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_radio_transmission(audio_path: str) -> dict:
    """
    Full pipeline: raw audio → preprocessed → transcribed → urgency classified → dispatched.
    """
    logger.info(f"Processing transmission: {audio_path}")

    # Step 1: Preprocess audio (denoise, normalise, segment)
    preprocessor = AudioPreprocessor(sample_rate=16000, n_mels=128)
    clean_audio, mel_spectrogram = preprocessor.process(audio_path)
    logger.info("Audio preprocessing complete.")

    # Step 2: CNN-based speech feature extraction & transcription
    cnn_model = CNNSpeechModel.load("models/cnn_speech_v2.pt")
    transcription, confidence = cnn_model.transcribe(mel_spectrogram)
    logger.info(f"Transcription: '{transcription}' (confidence: {confidence:.2f})")

    # Step 3: NLP urgency detection via semantic analysis
    detector = UrgencyDetector()
    urgency_result = detector.classify(transcription)
    logger.info(f"Urgency level: {urgency_result['level']} — {urgency_result['label']}")

    # Step 4: Dispatch based on urgency
    dispatcher = FirefighterDispatcher()
    dispatch_action = dispatcher.dispatch(urgency_result, transcription)

    return {
        "transcription": transcription,
        "confidence": confidence,
        "urgency": urgency_result,
        "dispatch": dispatch_action,
    }


if __name__ == "__main__":
    result = process_radio_transmission("samples/transmission_001.wav")
    print("\n=== TRANSMISSION REPORT ===")
    print(f"Text    : {result['transcription']}")
    print(f"Urgency : {result['urgency']['label']} (Level {result['urgency']['level']})")
    print(f"Action  : {result['dispatch']['action']}")
