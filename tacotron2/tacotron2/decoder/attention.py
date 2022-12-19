from dataclasses import dataclass
import torch
from torch import nn
import torch.functional as F

from tacotron2.layers import LinearNorm
from tacotron2.tacotron2.decoder.location_layer import LocationLayer


@dataclass
class AttentionConfig:
    attention_rnn_dim: int
    attention_dim: int
    attention_location_n_filters: int
    attention_location_kernel_size: int
    encoder_embedding_dim: int


class Attention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.query_layer = LinearNorm(
            config.attention_rnn_dim,
            config.attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        self.memory_layer = LinearNorm(
            config.encoder_embedding_dim,
            config.attention_dim,
            bias=False,
            w_init_gain='tanh'
        )
        self.v = LinearNorm(config.attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            config.attention_location_n_filters,
            config.attention_location_kernel_size,
            config.attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights
