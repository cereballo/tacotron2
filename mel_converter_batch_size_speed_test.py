from time import time

from torch.utils.data import DataLoader

from tacotron2.tacotron2.wav2mel_converter import Wav2MelConverter
from tacotron2.text_audio_collator import TextAudioCollator
from tacotron2.youtube_data import YouTubeData


DATASET_PATH = "/Users/smelsom/Code/audio-scraper/data/"
PHONEMIZER_PATH = "./data/models/deep_phonemizer/en_us_cmudict_ipa.pt"
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def main():
    dataset = YouTubeData(DATASET_PATH, PHONEMIZER_PATH)
    collator = TextAudioCollator(1)


    converter = Wav2MelConverter()

    for bs in BATCH_SIZES:
        start = time()
        data_loader = DataLoader(dataset, batch_size=bs, collate_fn=collator)

        dur_per_segment = []
        for _, _, wav_padded, _, _ in data_loader:
            start = time()
            converter.forward(wav_padded)
            dur_per_segment.append((time() - start) / bs)

        ave_dur_per_segment = sum(dur_per_segment) / len(dur_per_segment) / 1e-6
        print(f"Batch size: {bs}: {ave_dur_per_segment:.2f}us")

if __name__ == "__main__":
    main()
