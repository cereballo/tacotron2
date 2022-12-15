from dataclasses import dataclass

from tacotron2.tacotron2.config import Tacotron2Config


@dataclass
class Config:
    tacotron2_config: Tacotron2Config

    dataset_path: str

    epochs: int
    iters_per_checkpoint: int
    seed: int
    dynamic_loss_scaling: bool
    ignore_layers: list[str]

    text_cleaners: list[str]

    max_wav_value: int
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: int
    mel_fmax: int

    use_saved_learning_rate: bool
    learning_rate: float
    weight_decay: float
    grad_clip_thresh: float
    batch_size: int
    mask_padding: bool
