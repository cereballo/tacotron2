from dataclasses import dataclass

from tacotron2.tacotron2.decoder.decoder import DecoderConfig
from tacotron2.tacotron2.encoder import EncoderConfig
from tacotron2.tacotron2.postnet import PostNetConfig


@dataclass
class Tacotron2Config:
    encoder: EncoderConfig
    decoder: DecoderConfig
    postner: PostNetConfig

    n_symbols: int
    mask_padding: bool
    n_mel_channels: int
    n_frames_per_step: int
    symbols_embedding_dim: int

    def __init__(self, encoder: EncoderConfig, decoder: DecoderConfig, 
        postnet: PostNetConfig, n_symbols: int, symbols_embedding_dim: int,
        mask_padding: bool, n_mel_channels: int, n_frames_per_step: int):
        self.encoder = EncoderConfig(**encoder)
        self.decoder = DecoderConfig(**decoder)
        self.postnet = PostNetConfig(**postnet)
        self.n_symbols = n_symbols
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.symbols_embedding_dim = symbols_embedding_dim


@dataclass
class ExperimentConfig:
    epochs: int
    iters_per_checkpoint: int
    seed: int
    dynamic_loss_scaling: bool
    ignore_layers: list[str]
    output_dir: str
    checkpoint_path: str
    warm_start: bool
    dataset_path: str
    sampling_rate: int
    use_saved_learning_rate: bool
    n_frames_per_step: int
    pretrained_tacotron2_download_url: str
    pretrained_tacotron2_path: str
    pretrained_phonemizer_download_url: str
    pretrained_phonemizer_path: str


@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float
    grad_clip_thresh: float


@dataclass
class Config:
    experiment: ExperimentConfig
    tacotron2: Tacotron2Config
    optimizer: OptimizerConfig

    def __init__(self, experiment: ExperimentConfig, tacotron2: Tacotron2Config, optimizer: OptimizerConfig):
        self.experiment = ExperimentConfig(**experiment)
        self.tacotron2 = Tacotron2Config(**tacotron2)
        self.optimizer = OptimizerConfig(**optimizer)
