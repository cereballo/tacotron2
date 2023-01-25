import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    if torch.has_cuda:
        device = "cuda"
    elif torch.has_mps:
        device = "mps"
    else:
        device = "cpu"
    ids = torch.arange(0, max_len, out=torch.empty(max_len, dtype=torch.long, device=device))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    elif torch.has_mps:
        x = x.to(torch.device("mps"))
    return torch.autograd.Variable(x)
