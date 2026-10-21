"""Federated full-genome CNN training with a small, configurable sample limit."""

import argparse
import math
import os
import statistics
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


def training_score_stats(rows):
    """Fit normalization to selected training scores only; handle constant targets."""
    scores = [score for _, score in rows]
    mean = statistics.mean(scores)
    scale = statistics.pstdev(scores)
    return mean, scale if scale > 1e-6 else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--score_key", default="Score")
    parser.add_argument("--num_samples", type=positive_int, default=160)
    parser.add_argument("--eval_samples", type=positive_int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    device = select_device(args.device)
    flare.init()
    site_name = flare.get_site_name()
    start = time.perf_counter()
    train_loader, validation_loader = create_loaders(args.repo, args.num_samples, args.eval_samples, args.score_key)
    score_mean, score_scale = training_score_stats(train_loader.dataset)
    baseline_mse = sum((score - score_mean) ** 2 for _, score in validation_loader.dataset) / len(
        validation_loader.dataset
    )
    print(
        f"{site_name}: loaded {len(train_loader.dataset)} training and {len(validation_loader.dataset)} "
        f"evaluation genomes in {time.perf_counter() - start:.2f}s; device={device}, threads={threads}",
        flush=True,
    )
    print(
        f"{site_name}: training score mean={score_mean:.4f}, scale={score_scale:.4f}; "
        f"held-out training-mean baseline MSE={baseline_mse:.4f}",
        flush=True,
    )
    model = create_model().to(device)
    while flare.is_running():
        incoming = flare.receive()
        model.load_state_dict(incoming.params)
        synchronize(device)
        start = time.perf_counter()
        mse = evaluate(model, validation_loader, device)
        synchronize(device)
        print(
            f"{site_name}: evaluation={time.perf_counter() - start:.2f}s, MSE={mse:.4f}, "
            f"baseline MSE={baseline_mse:.4f}, improvement={baseline_mse - mse:.4f}",
            flush=True,
        )
        metrics = {"mse": mse, "baseline_mse": baseline_mse}
        if flare.is_evaluate():
            flare.send(flare.FLModel(metrics=metrics))
            continue
        model.train()
        model.normalize_output(score_mean, score_scale)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        synchronize(device)
        start = time.perf_counter()
        normalized_squared_error = 0.0
        for bases, score in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = torch.tensor((score - score_mean) / score_scale, dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(model(bases.to(device)), target)
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite training loss")
            normalized_squared_error += loss.detach().item()
            loss.backward()
            optimizer.step()
        model.restore_output(score_mean, score_scale)
        synchronize(device)
        seconds = time.perf_counter() - start
        steps = len(train_loader)
        train_mse = normalized_squared_error / steps * score_scale ** 2
        if not math.isfinite(train_mse):
            raise ValueError("Nonfinite training MSE")
        print(
            f"{site_name}: round={incoming.current_round}, training={seconds:.2f}s, "
            f"mean/genome={seconds / steps:.3f}s, online training MSE={train_mse:.4f}",
            flush=True,
        )
        flare.send(
            flare.FLModel(
                params={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                metrics=metrics,
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    main()
