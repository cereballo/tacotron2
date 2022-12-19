from pathlib import Path
import argparse

import gdown
import pytorch_lightning as pl
import toml
import torch
from torch.utils.data import DataLoader

from tacotron2.tacotron2.tacotron2 import Tacotron2
from tacotron2.config import Config
from tacotron2.datasets.youtube import YouTube


def download_pretrained_model():
    url = 'https://drive.google.com/u/0/uc?id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA&export=download'
    download_dir = Path("data", "models", "nvidia_pretrained_model")
    download_dir.mkdir(parents=True, exist_ok=True)
    filepath = download_dir / "tacotron2_statedict.pt"
    if not filepath.exists():
        gdown.download(url, str(filepath))


def warm_start_model(checkpoint_path, model, ignore_layers):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path("config.toml"))
    return parser.parse_args()


def main():
    args = parse_args()
    config_dict = toml.loads(args.config.read_text())
    config = Config(**config_dict)

    train_loader = DataLoader(YouTube(config.experiment.dataset_path))

    download_pretrained_model()

    # trainer = pl.Trainer()
    # trainer.fit(Tacotron2, train_loader)


if __name__ == '__main__':
    main()
