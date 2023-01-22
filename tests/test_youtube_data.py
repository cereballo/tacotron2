from loguru import logger as log
import pytest

from tacotron2.youtube_data import YouTubeData

DATASET_PATH = "/Users/smelsom/Code/audio-scraper/data/"
PHONEMIZER_PATH = "./data/models/deep_phonemizer/en_us_cmudict_ipa.pt"


@pytest.fixture
def data_loader():
    yield YouTubeData(DATASET_PATH, PHONEMIZER_PATH)


class TestYouTubeData:

    def test_get_item(self, data_loader: YouTubeData):
        phones, wavs = data_loader.__getitem__(0)
        log.info(f"Number of phones: {len(phones)}")
        log.info(f"Number of audio samples: {len(wavs)}")
        assert len(phones) > 0
        assert len(wavs) > 0
