from dataclasses import dataclass
import torch
from torch import nn
from torchaudio.transforms import Resample, Spectrogram, MelScale


@dataclass
class DataHeadConfig:
    input_sampling_rate: int
    output_sampling_rate: int
    n_fft: int
    n_mel: int


class DataHead(nn.Module):
    def __init__(
        self,
        input_freq=16000,
        resample_freq=8000,
        n_fft=1024,
        n_mel=80
    ):
        super().__init__()
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

        self.spec = Spectrogram(n_fft=n_fft, power=2)

        self.mel_scale = MelScale(
            n_mels=n_mel, 
            sample_rate=resample_freq, 
            n_stft=n_fft // 2 + 1
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        resampled = self.resample(waveform)

        # Convert to power spectrogram
        spec = self.spec(resampled)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        return mel
