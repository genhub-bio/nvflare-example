"""Run two clients with independently configured organization Gen repositories."""

import argparse
import os
import shlex
import time
from pathlib import Path

import torch
from client import positive_int
from model import create_model, model_spec

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_final_global_evaluation, set_per_site_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_a_repo", type=Path, default=Path(__file__).resolve().parent / "organization-a")
    parser.add_argument("--org_b_repo", type=Path, default=Path(__file__).resolve().parent / "organization-b")
    parser.add_argument("--num_rounds", type=positive_int, default=1)
    parser.add_argument("--num_samples", type=positive_int, default=160, help="Training genomes per client")
    parser.add_argument("--eval_samples", type=positive_int, default=40, help="Held-out genomes per client")
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--channels", type=positive_int, default=8)
    parser.add_argument("--chunk_size", type=positive_int, default=65536)
    parser.add_argument("--threads", type=positive_int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    spec = model_spec(args.channels, args.chunk_size)
    train_args = (
        f" --num_samples {args.num_samples} --eval_samples {args.eval_samples} --epochs {args.epochs}"
        f" --channels {args.channels} --chunk_size {args.chunk_size} --threads {args.threads}"
        f" --device {args.device}"
    )
    recipe = FedAvgRecipe(
        name="multiple-organizations",
        model=create_model(args.channels, args.chunk_size),
        min_clients=2,
        num_rounds=args.num_rounds,
        train_script="client.py",
        key_metric="mse",
        key_metric_mode="min",
    )
    set_per_site_config(
        recipe,
        {
            # Use strings for compatibility with installed NVFlare releases.
            "site-1": {"train_args": f"--repo {shlex.quote(str(args.org_a_repo.expanduser().resolve()))}{train_args}"},
            "site-2": {"train_args": f"--repo {shlex.quote(str(args.org_b_repo.expanduser().resolve()))}{train_args}"},
        },
    )
    add_final_global_evaluation(recipe)
    start = time.perf_counter()
    run = recipe.execute(SimEnv(clients=["site-1", "site-2"]))
    result = run.get_result(clean_up=False)
    print(f"Federated run wall time: {time.perf_counter() - start:.2f}s", flush=True)
    if result is None:
        raise RuntimeError("Simulation did not return a result workspace.")
    checkpoint = Path(result) / "server" / "simulate_job" / "app_server" / "FL_global_model.pt"
    output = Path(__file__).resolve().parent / "models" / "global_model.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved["model_spec"] = spec
    temporary = output.with_suffix(".part")
    try:
        torch.save(saved, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print("Results:", result)
    print("Saved model:", output)


if __name__ == "__main__":
    main()
