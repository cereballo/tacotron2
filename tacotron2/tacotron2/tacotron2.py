from math import sqrt

import pytorch_lightning as pl
import torch
from torch import nn
from tacotron2.config import OptimizerConfig, Tacotron2Config

from tacotron2.loss_function import Tacotron2Loss
from tacotron2.tacotron2.data_head import DataHead
from tacotron2.tacotron2.decoder.decoder import Decoder
from tacotron2.tacotron2.encoder import Encoder
from tacotron2.tacotron2.postnet import PostNet
from tacotron2.utils import get_mask_from_lengths, to_gpu




class Tacotron2(pl.LightningModule):

    opt_config: OptimizerConfig

    def __init__(self, config: Tacotron2Config, opt_config: OptimizerConfig):
        super(Tacotron2, self).__init__()
        self.mask_padding = config.mask_padding
        self.n_mel_channels = config.n_mel_channels
        self.n_frames_per_step = config.n_frames_per_step
        self.embedding = nn.Embedding(config.n_symbols, config.symbols_embedding_dim)
        std = sqrt(2.0 / (config.n_symbols + config.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.opt_config = opt_config

        self.data_head = DataHead()
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
        self.postnet = PostNet(config.postnet)
        self.loss = Tacotron2Loss()

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def _forward(self, inputs):
        text_inputs, text_lengths, wavs, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
       
        mels = self.data_head(wavs)

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def training_step(self, batch, batch_idx):
        x, y = self.parse_batch(batch)
        y_pred = self._forward(x)
        return self.loss(y_pred, y)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.opt_config.learning_rate,
            weight_decay=self.opt_config.weight_decay
        )

    def __call__(self, inputs):
        """
        Infer from the model.
        """
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
