"""Full-sequence CNN and shared encoding for federated training and prediction."""

import numpy as np
import torch
from data import dna_sequence
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODEL_VERSION = "full-sequence-cnn-v2"


class FullSequenceRegressor(nn.Module):
    """Learn local DNA patterns, pool over every base, and predict one genome score."""

    # Two stride-one convolutions with kernel sizes 9 and 5 have a 6-base halo.
    HALO = 6

    def __init__(self, channels=8, chunk_size=65536):
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder = nn.Sequential(
            nn.Conv1d(4, channels, 9, padding=4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.output = nn.Linear(channels, 1)

    @torch.no_grad()
    def normalize_output(self, mean, scale):
        """Convert a raw-score head to local standardized-score units before training."""
        self.output.weight.div_(scale)
        self.output.bias.sub_(mean).div_(scale)

    @torch.no_grad()
    def restore_output(self, mean, scale):
        """Return the head to raw-score units before evaluation, aggregation, or saving."""
        self.output.weight.mul_(scale)
        self.output.bias.mul_(scale).add_(mean)

    def chunk_sum(self, bases, begin, end):
        one_hot = F.one_hot(bases.long(), num_classes=4).float().T.unsqueeze(0)
        return self.encoder(one_hot)[..., begin:end].sum(dim=-1)

    def forward(self, bases):
        if bases.ndim != 1 or not bases.numel():
            raise ValueError("Expected one nonempty encoded genome")
        length = bases.numel()
        pooled = None
        for begin in range(0, length, self.chunk_size):
            end = min(begin + self.chunk_size, length)
            left, right = max(0, begin - self.HALO), min(length, end + self.HALO)
            args = (bases[left:right], begin - left, end - left)
            if self.training and torch.is_grad_enabled():
                # Recompute chunk activations during backward to bound training memory.
                value = checkpoint(self.chunk_sum, *args, use_reentrant=False)
            else:
                value = self.chunk_sum(*args)
            pooled = value if pooled is None else pooled + value
        return self.output(pooled / length).reshape(())


def encode_sequence(sequence):
    sequence = dna_sequence(sequence)
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[list(b"ACGT")] = np.arange(4, dtype=np.uint8)
    return torch.from_numpy(lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)])


def model_spec(channels=8, chunk_size=65536):
    if type(channels) is not int or type(chunk_size) is not int or min(channels, chunk_size) < 1:
        raise ValueError("channels and chunk_size must be positive integers")
    return {"version": MODEL_VERSION, "channels": channels, "chunk_size": chunk_size}


def create_model(channels=8, chunk_size=65536):
    model_spec(channels, chunk_size)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return FullSequenceRegressor(channels, chunk_size)


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
