"""Federated full-genome CNN training with a small, configurable sample limit."""

import argparse
import os
import time

import torch
from data import open_repository, sample_rows, split_samples
from model import create_model, encode_sequence, select_device, synchronize
from torch.utils.data import DataLoader

import nvflare.client as flare


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def create_loaders(path, num_samples=160, eval_samples=40, score_key="Score"):
    repo = open_repository(path)
    train, validation = split_samples(repo, score_key)
    if not 1 <= num_samples <= len(train) or not 1 <= eval_samples <= len(validation):
        raise ValueError(
            f"Requested {num_samples} training/{eval_samples} evaluation samples; "
            f"available: {len(train)}/{len(validation)}"
        )

    def dataset(samples):
        return [(encode_sequence(sequence), score) for sequence, score in sample_rows(samples, score_key)]

    return (
        DataLoader(
            dataset(train[:num_samples]), batch_size=None, shuffle=True, generator=torch.Generator().manual_seed(42)
        ),
        DataLoader(dataset(validation[:eval_samples]), batch_size=None),
    )


def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    with torch.no_grad():
        for bases, score in loader:
            squared_error += (model(bases.to(device)).item() - score) ** 2
    return squared_error / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--score_key", default="Score")
    parser.add_argument("--num_samples", type=positive_int, default=160)
    parser.add_argument("--eval_samples", type=positive_int, default=40)
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--channels", type=positive_int, default=8)
    parser.add_argument("--chunk_size", type=positive_int, default=65536)
    parser.add_argument("--threads", type=positive_int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    flare.init()
    site_name = flare.get_site_name()
    start = time.perf_counter()
    train_loader, validation_loader = create_loaders(args.repo, args.num_samples, args.eval_samples, args.score_key)
    print(
        f"{site_name}: loaded {len(train_loader.dataset)} training and {len(validation_loader.dataset)} "
        f"evaluation genomes in {time.perf_counter() - start:.2f}s; device={device}, threads={args.threads}",
        flush=True,
    )
    model = create_model(args.channels, args.chunk_size).to(device)
    while flare.is_running():
        incoming = flare.receive()
        model.load_state_dict(incoming.params)
        synchronize(device)
        start = time.perf_counter()
        mse = evaluate(model, validation_loader, device)
        synchronize(device)
        print(f"{site_name}: evaluation={time.perf_counter() - start:.2f}s, MSE={mse:.4f}", flush=True)
        if flare.is_evaluate():
            flare.send(flare.FLModel(metrics={"mse": mse}))
            continue
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.epochs):
            for bases, score in train_loader:
                optimizer.zero_grad(set_to_none=True)
                target = torch.tensor(score, dtype=torch.float32, device=device)
                loss = torch.nn.functional.mse_loss(model(bases.to(device)), target)
                loss.backward()
                optimizer.step()
        synchronize(device)
        seconds = time.perf_counter() - start
        steps = args.epochs * len(train_loader)
        print(
            f"{site_name}: round={incoming.current_round}, training={seconds:.2f}s, "
            f"mean/genome={seconds / steps:.3f}s",
            flush=True,
        )
        flare.send(
            flare.FLModel(
                params={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                metrics={"mse": mse},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    main()
