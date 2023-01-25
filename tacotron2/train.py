import argparse
from pathlib import Path

from loguru import logger as log
import gdown
import pytorch_lightning as pl
import requests
import toml
import torch
from torch.utils.data import DataLoader
from tacotron2.text_mel_collator import TextMelCollator

from tacotron2.config import Config
from tacotron2.youtube_data import YouTubeData
from tacotron2.ljspeech_data import LJSPEECH
from tacotron2.tacotron2.tacotron2 import Tacotron2


def download_pretrained_model(url: str, dest: Path, site: str = "other"):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if site == "google_drive":
        log.info(f"Downloding pretrained model to: {dest}")
        if not dest.exists():
            gdown.download(url, str(dest))
    else:
        log.info("Pretrained model already downloaded.")
        download = requests.get(url)
        dest.write_bytes(download.content)


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
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--mps", action="store_true", help="Enable MPS acceleration (M1 Mac only)")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dict = toml.loads(args.config.read_text())
    config = Config(**config_dict)

    log.info(f"Training using config: {config}")

    download_pretrained_model(
        config.experiment.pretrained_phonemizer_download_url,
        Path(config.experiment.pretrained_phonemizer_path)
    )

    log.info("Preparing dataset...")
    
    if config.experiment.dataset == "youtube":
        train_dataset = YouTubeData(
            config.experiment.dataset_path,
            config.experiment.pretrained_phonemizer_path
        )
    elif config.experiment.dataset == "ljspeech":
        train_dataset = LJSPEECH(
            "./data",
            config.experiment.pretrained_phonemizer_path,
            download=True
        )
    else:
        raise NotImplementedError(f"Dataset: {config.experiment.dataset} has not been implemented for training.")

    data_collator = TextMelCollator(config.experiment.n_frames_per_step)
    train_loader = DataLoader(train_dataset, collate_fn=data_collator, num_workers=2)

    download_pretrained_model(
        config.experiment.pretrained_tacotron2_download_url,
        Path(config.experiment.pretrained_tacotron2_path),
        "google_drive"
    )

    log.info("Loading model...")
    tt2 = Tacotron2(config.tacotron2, config.optimizer)

    if args.mps:
        trainer = pl.Trainer(accelerator="mps", devices=1)
    else:
        trainer = pl.Trainer()

    trainer.fit(model=tt2, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()
