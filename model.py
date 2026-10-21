"""Full-sequence CNN and shared encoding for federated training and prediction."""

import numpy as np
import torch
from data import dna_sequence
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODEL_VERSION = "full-sequence-cnn-v4"
DEFAULT_CHANNELS = 10
DEFAULT_REGIONS = 100


class FullSequenceRegressor(nn.Module):
    """Learn local DNA patterns, pool over every base, and predict one genome score."""

    # The second convolution needs two neighboring 100-base windows on each side.
    WINDOW_SIZE = 100
    FEATURE_HALO = 2

    def __init__(self, channels=DEFAULT_CHANNELS, chunk_size=65536, regions=DEFAULT_REGIONS):
        super().__init__()
        self.chunk_size = chunk_size
        self.regions = regions
        self.encoder = nn.Sequential(
            nn.Conv1d(4, channels, self.WINDOW_SIZE, stride=self.WINDOW_SIZE),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.output = nn.Linear(channels * regions, 1)

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
        """Sum selected feature positions from an input aligned to 100-base windows."""
        one_hot = F.one_hot(bases.long(), num_classes=4).float().T.unsqueeze(0)
        one_hot = F.pad(one_hot, (0, (-bases.numel()) % self.WINDOW_SIZE))
        return self.encoder(one_hot)[..., begin:end].sum(dim=-1)

    def forward(self, bases):
        if bases.ndim != 1 or not bases.numel():
            raise ValueError("Expected one nonempty encoded genome")
        length = bases.numel()
        windows = (length + self.WINDOW_SIZE - 1) // self.WINDOW_SIZE
        chunk_windows = max(1, self.chunk_size // self.WINDOW_SIZE)
        regional_features = []
        for region in range(self.regions):
            start, stop = region * windows // self.regions, (region + 1) * windows // self.regions
            if start == stop:
                # Keep a fixed shape when there are fewer output windows than regions.
                regional_features.append(self.output.weight.new_zeros(1, self.encoder[0].out_channels))
                continue
            pooled = None
            for begin in range(start, stop, chunk_windows):
                end = min(begin + chunk_windows, stop)
                # Keep window alignment and context across chunk and region boundaries.
                left = max(0, begin - self.FEATURE_HALO)
                right = min(windows, end + self.FEATURE_HALO)
                args = (
                    bases[left * self.WINDOW_SIZE:min(length, right * self.WINDOW_SIZE)],
                    begin - left,
                    end - left,
                )
                if self.training and torch.is_grad_enabled():
                    # Recompute chunk activations during backward to bound training memory.
                    value = checkpoint(self.chunk_sum, *args, use_reentrant=False)
                else:
                    value = self.chunk_sum(*args)
                pooled = value if pooled is None else pooled + value
            regional_features.append(pooled / (stop - start))
        return self.output(torch.cat(regional_features, dim=-1)).reshape(())


def encode_sequence(sequence):
    sequence = dna_sequence(sequence)
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[list(b"ACGT")] = np.arange(4, dtype=np.uint8)
    return torch.from_numpy(lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)])


def model_spec(channels=DEFAULT_CHANNELS, chunk_size=65536, regions=DEFAULT_REGIONS):
    if any(type(value) is not int or value < 1 for value in (channels, chunk_size, regions)):
        raise ValueError("channels, chunk_size, and regions must be positive integers")
    return {"version": MODEL_VERSION, "channels": channels, "chunk_size": chunk_size, "regions": regions}


def create_model(channels=DEFAULT_CHANNELS, chunk_size=65536, regions=DEFAULT_REGIONS):
    model_spec(channels, chunk_size, regions)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return FullSequenceRegressor(channels, chunk_size, regions)


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
