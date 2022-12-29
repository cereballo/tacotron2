import torch


class TextAudioCollator():
    """
    Zero-pads model inputs and targets based on number of frames per step.
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """
        Collate's training batch from normalized text and mel-spectrogram
        
        @input: batch: [text_normalized, audio]
        """
        # Right zero-pad all one-hot text sequences to max input length
        text_lens = torch.LongTensor([len(x[0]) for x in batch])
        input_lengths, ids_sorted_decreasing = torch.sort(text_lens, dim=0, descending=True)
        max_text_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_text_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad waveform
        max_target_len = max([x[1].size(0) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include wav padded and gate padded
        wav_padded = torch.FloatTensor(len(batch), max_target_len)
        wav_padded.zero_()

        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            wav = batch[ids_sorted_decreasing[i]][1]
            wav_padded[i, :wav.size(0)] = wav
            gate_padded[i, wav.size(0)-1:] = 1
            output_lengths[i] = wav.size(0)

        return text_padded, input_lengths, wav_padded, gate_padded, output_lengths
