"""
CNNSpeechModel
--------------
Convolutional Neural Network for extracting phoneme-level features
from mel spectrograms and producing word-level transcriptions.

Architecture:
  Input mel spectrogram (1 × 128 × T)
    → Conv Block 1: 32 filters, 3×3, BN, ReLU, MaxPool
    → Conv Block 2: 64 filters, 3×3, BN, ReLU, MaxPool
    → Conv Block 3: 128 filters, 3×3, BN, ReLU, MaxPool
    → Global Average Pool
    → Dense 256 → Dropout 0.4
    → CTC output layer (vocabulary size)

The CNN captures local spectro-temporal patterns (formants, transients)
that are robust to background noise found in fire environments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class CNNSpeechModel(nn.Module):
    VOCAB_SIZE = 29  # a-z + space + blank + <eos>

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))  # keep time axis
        self.dense = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.output = nn.Linear(256, self.VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)          # (batch, 128, 1, time')
        x = x.squeeze(2).permute(0, 2, 1)  # (batch, time', 128)
        x = self.dense(x)
        x = self.output(x)               # (batch, time', vocab)
        return torch.log_softmax(x, dim=-1)

    @classmethod
    def load(cls, checkpoint_path: str) -> "CNNSpeechModel":
        model = cls()
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model

    def transcribe(self, mel_spectrogram: np.ndarray) -> Tuple[str, float]:
        """
        Run inference on a single mel spectrogram.
        Returns (transcription_text, confidence_score).
        """
        tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.forward(tensor)           # (1, T, vocab)
        probs = log_probs.exp()
        best_path = probs.argmax(dim=-1).squeeze(0)    # greedy CTC decode
        confidence = float(probs.max(dim=-1).values.mean())
        transcription = self._ctc_decode(best_path.tolist())
        return transcription, confidence

    def _ctc_decode(self, indices: list) -> str:
        """Greedy CTC collapse: remove blanks and repeated tokens."""
        BLANK = self.VOCAB_SIZE - 1
        CHARS = "abcdefghijklmnopqrstuvwxyz "
        result, prev = [], None
        for idx in indices:
            if idx != BLANK and idx != prev:
                if idx < len(CHARS):
                    result.append(CHARS[idx])
            prev = idx
        return "".join(result).strip()
