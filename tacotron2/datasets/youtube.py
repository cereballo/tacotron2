from pathlib import Path

from torch.utils.data import Dataset
import torchaudio


class YouTube(Dataset):
    """
    Loads data from audio-scraper dataset.
    """

    transcript_paths: list[Path]
    chunk_paths: list[Path]

    def __init__(self, dataset_path: str):
        chunks_path = Path(dataset_path) / "chunks"
        if not chunks_path.exists():
            raise FileNotFoundError("Chunks path, %s, is not valid." % chunks_path)

        transcripts_path = Path(dataset_path) / "transcripts"
        if not transcripts_path.exists():
            raise FileNotFoundError("Transcripts, %s, path is not valid." % transcripts_path)
            
        channels = transcripts_path.glob("*")

        self.transcript_paths = []
        self.chunk_paths = []
        for channel in channels:
            self.transcript_paths.extend((transcripts_path / channel.name).glob("*"))
            self.chunk_paths.extend((chunks_path / channel.name).glob("*"))

    def __getitem__(self, idx):
        transcript = self.transcript_paths[idx].read_text()
        chunk = torchaudio.load(self.chunk_paths[idx])
        return chunk, transcript

    def __len__(self):
        return len(self.transcript_paths)


def main():
    ds = YouTube("/Users/smelsom/Code/audio-scraper/data/")
    i = ds.__getitem__(0)
    print(i)


if __name__ == "__main__":
    main()
