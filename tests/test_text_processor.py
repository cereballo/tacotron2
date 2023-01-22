import pytest

from tacotron2.text_processor import TextProcessor

TEXT_PROCESSOR_PATH = "./data/models/deep_phonemizer/en_us_cmudict_ipa.pt"


@pytest.fixture
def text_processor():
    yield TextProcessor(TEXT_PROCESSOR_PATH)


class TestTextProcessor:

    def test_call(self, text_processor):
        phones = text_processor("hello")
        assert phones is not None
