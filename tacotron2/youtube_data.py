from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F

from tacotron2.text_processor import TextProcessor


class YouTubeData(Dataset):
    """
    Loads data from audio-scraper dataset.
    """

    transcript_paths: list[Path]
    chunk_paths: list[Path]
    text_processor: TextProcessor

    def __init__(self, dataset_path: str, phonemizer_path: str):
        chunks_path = Path(dataset_path) / "chunks"
        if not chunks_path.exists():
            raise FileNotFoundError("Chunks path, %s, is not valid." % chunks_path)

        transcripts_path = Path(dataset_path) / "transcripts"
        if not transcripts_path.exists():
            raise FileNotFoundError("Transcripts, %s, path is not valid." % transcripts_path)
            
        yt_channels = transcripts_path.glob("*")

        self.transcript_paths = []
        self.chunk_paths = []
        for channel in yt_channels:
            self.transcript_paths.extend((transcripts_path / channel.name).glob("*"))
            self.chunk_paths.extend((chunks_path / channel.name).glob("*"))

        self.text_processor = TextProcessor(phonemizer_path)

    def __getitem__(self, idx):
        transcript = self.transcript_paths[idx].read_text().strip()
        phone_tensors = self._preprocess_text(transcript)
        chunk, _ = torchaudio.load(self.chunk_paths[idx])
        return phone_tensors, chunk[0]

    def _preprocess_text(self, text: str):
        return torch.LongTensor(self.text_processor(text))
        # return F.one_hot(symbol_ids, num_classes=self.text_processor.n_symbols)

    def __len__(self):
        return len(self.transcript_paths)


def main():
    ds = YouTubeData("/Users/smelsom/Code/audio-scraper/data/", "en_us_cmudict_ipa.pt")
    i = ds.__getitem__(0)
    print(i)


if __name__ == "__main__":
    main()