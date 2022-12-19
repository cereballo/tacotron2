from dataclasses import dataclass

from tacotron2.tacotron2.encoder import EncoderConfig
from tacotron2.tacotron2.decoder.decoder import DecoderConfig
from tacotron2.tacotron2.postnet import PostNetConfig

@dataclass
class Tacotron2Config:
    encoder: EncoderConfig
    decoder: DecoderConfig
    postner: PostNetConfig

    n_symbols: int
    symbols_embedding_dim: int

    def __init__(self, encoder: EncoderConfig, decoder: DecoderConfig, postnet: PostNetConfig, n_symbols: int, symbols_embedding_dim: int):
        self.encoder = EncoderConfig(**encoder)
        self.decoder = DecoderConfig(**decoder)
        self.postnet = PostNetConfig(**postnet)
        self.n_symbols = n_symbols
        self.symbols_embedding_dim = symbols_embedding_dim
