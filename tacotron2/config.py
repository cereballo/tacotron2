from dataclasses import dataclass

from tacotron2.tacotron2.config import Tacotron2Config


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


@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float
    grad_clip_thresh: float
    batch_size: int
    mask_padding: bool

@dataclass
class Config:
    experiment: ExperimentConfig
    tacotron2: Tacotron2Config
    optimizer: OptimizerConfig

    def __init__(self, experiment: ExperimentConfig, tacotron2: Tacotron2Config, optimizer: OptimizerConfig):
        self.experiment = ExperimentConfig(**experiment)
        self.tacotron2 = Tacotron2Config(**tacotron2)
        self.optimizer = OptimizerConfig(**optimizer)
