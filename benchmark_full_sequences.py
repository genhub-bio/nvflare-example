"""Time local CNN training on a small number of complete Gen sample sequences."""

import argparse
import os
import time
from pathlib import Path

import torch
from data import open_repository, sample_rows, split_samples
from model import FullSequenceRegressor, encode_sequence
from torch.nn import functional as F


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent / "organization-a")
    parser.add_argument("--num_samples", type=positive_int, default=5, help="Total training genomes, not per client")
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--chunk_size", type=positive_int, default=65536)
    parser.add_argument("--channels", type=positive_int, default=8)
    parser.add_argument("--threads", type=positive_int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--score_key", default="Score")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    device = torch.device(args.device)
    start = time.perf_counter()
    repo = open_repository(args.repo)
    training, _ = split_samples(repo, args.score_key)
    if args.num_samples > len(training):
        parser.error(f"Requested {args.num_samples} genomes but the training split contains only {len(training)}")
    selected = training[: args.num_samples]
    rows = [(encode_sequence(sequence), score) for sequence, score in sample_rows(selected, args.score_key)]
    loading_seconds = time.perf_counter() - start
    model = FullSequenceRegressor(args.channels, args.chunk_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    print(f"device={device}, torch={torch.__version__}, CPU threads={args.threads}, batch_size=1", flush=True)
    print(
        f"channels={args.channels}, chunk_size={args.chunk_size}, parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    print(f"Loaded and encoded {len(rows)} complete genomes in {loading_seconds:.2f}s", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
    step_times = []
    model.train()
    synchronize(device)
    training_start = time.perf_counter()
    for epoch in range(args.epochs):
        for sample, (bases, score) in zip(selected, rows):
            synchronize(device)
            start = time.perf_counter()
            bases = bases.to(device)
            target = torch.tensor(score, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(bases)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            synchronize(device)
            seconds = time.perf_counter() - start
            step_times.append(seconds)
            print(
                f"epoch={epoch + 1} sample={sample.name} bases={bases.numel()} "
                f"train_step={seconds:.3f}s loss={loss.item():.4f}",
                flush=True,
            )
    synchronize(device)
    training_seconds = time.perf_counter() - training_start
    print(
        f"Training total={training_seconds:.2f}s, mean/genome={sum(step_times) / len(step_times):.3f}s, "
        f"bases/s={sum(b.numel() for b, _ in rows) * args.epochs / training_seconds:,.0f}",
        flush=True,
    )
    if len(step_times) > 1:
        print(f"Mean/genome excluding first step={sum(step_times[1:]) / len(step_times[1:]):.3f}s", flush=True)
    if device.type == "cuda":
        print(f"Peak CUDA tensor memory={torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
