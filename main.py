"""
AudioPreprocessor
-----------------
Handles denoising, normalisation, and mel-spectrogram extraction
for high-noise fire emergency radio transmissions.
"""

import numpy as np
import librosa
import noisereduce as nr
from typing import Tuple


class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 512, hop_length: int = 160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, path: str) -> np.ndarray:
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        return audio

    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction-based noise reduction.
        Uses the first 0.5s of audio as a noise profile estimate.
        """
        noise_sample = audio[: int(self.sample_rate * 0.5)]
        reduced = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=self.sample_rate, prop_decrease=0.85)
        return reduced

    def normalise(self, audio: np.ndarray) -> np.ndarray:
        """Peak normalisation to [-1, 1]."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute log-mel spectrogram — the primary input feature for the CNN.
        Shape: (n_mels, time_frames)
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=8000,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel

    def process(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.
        Returns: (clean_audio, mel_spectrogram)
        """
        audio = self.load_audio(audio_path)
        audio = self.reduce_noise(audio)
        audio = self.normalise(audio)
        audio = self.trim_silence(audio)
        mel = self.extract_mel_spectrogram(audio)
        return audio, mel
