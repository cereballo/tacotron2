from dataclasses import dataclass

from tacotron2.tacotron2.encoder import EncoderConfig
from tacotron2.tacotron2.decoder.decoder import DecoderConfig
from tacotron2.tacotron2.postnet import PostNetConfig

@dataclass
class Tacotron2Config:
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig
    postnet_config: PostNetConfig

    n_symbols: int
    symbols_embedding_dim: int
