import csv
import os
from pathlib import Path

import torch
import torchaudio
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive
from tacotron2.tacotron2.wav2mel_converter import Wav2MelConverter

from tacotron2.text_processor import TextProcessor


_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "wavs",
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "checksum": "be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5",
    }
}


class LJSPEECH(Dataset):
    """*LJSpeech-1.1* :cite:`ljspeech17` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"wavs"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """
    text_processor: TextProcessor
    wav2mel_converter: Wav2MelConverter

    def __init__(
        self,
        root: str | Path,
        phonemizer_path: str,
        url: str = _RELEASE_CONFIGS["release1"]["url"],
        folder_in_archive: str = _RELEASE_CONFIGS["release1"]["folder_in_archive"],
        download: bool = False,
    ) -> None:

        self._parse_filesystem(root, url, folder_in_archive, download)

        self.text_processor = TextProcessor(phonemizer_path)

        self.wav2mel_converter = Wav2MelConverter()

    def _parse_filesystem(self, root: str | Path, url: str, folder_in_archive: str | Path, download: bool) -> None:
        root = Path(root)

        basename = os.path.basename(url)
        archive = root / basename

        basename = Path(basename.split(".tar.bz2")[0])
        folder_in_archive = basename / folder_in_archive

        self._path = root / folder_in_archive
        self._metadata_path = root / basename / "metadata.csv"

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _RELEASE_CONFIGS["release1"]["checksum"]
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive)
        else:
            if not os.path.exists(self._path):
                raise RuntimeError(
                    f"The path {self._path} doesn't exist. "
                    "Please check the ``root`` path or set `download=True` to download it"
                )

        with open(self._metadata_path, "r", newline="") as metadata:
            flist = csv.reader(metadata, delimiter="|", quoting=csv.QUOTE_NONE)
            self._flist = list(flist)

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            str:
                Normalized Transcript
        """
        line = self._flist[n]
        fileid, transcript, normalized_transcript = line
        fileid_audio = self._path / (fileid + ".wav")

        # Load audio
        waveform, _ = torchaudio.load(fileid_audio)
        mel = self.wav2mel_converter(waveform[0])

        phone_tensors = self._preprocess_text(transcript)

        return phone_tensors, mel

    def _preprocess_text(self, text: str):
        return torch.LongTensor(self.text_processor(text))

    def __len__(self) -> int:
        return len(self._flist)


def main():
    ds = LJSPEECH("/Users/smelsom/Code/audio-scraper/data/", "en_us_cmudict_ipa.pt")
    i = ds.__getitem__(0)
    print(i)

if __name__ == "__main__":
    main()
