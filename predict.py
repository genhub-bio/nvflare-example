"""Load the saved global model and score sequences from a FASTA or Gen GFA file on the CPU."""

import argparse
from pathlib import Path

import torch
from data import read_fasta, read_gfa
from model import MODEL_VERSION, create_model, encode_sequence, model_spec

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "global_model.pt"


def load_model(path=DEFAULT_MODEL_PATH):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    spec = checkpoint.get("model_spec")
    if not isinstance(spec, dict) or spec.get("version") != MODEL_VERSION:
        raise ValueError("Checkpoint is not a full-sequence CNN; retrain with job.py")
    if spec != model_spec(spec.get("channels"), spec.get("chunk_size")):
        raise ValueError("Unsupported model configuration")
    model = create_model(spec["channels"], spec["chunk_size"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict(model, sequences):
    with torch.inference_mode():
        return [model(encode_sequence(sequence)).item() for sequence in sequences]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="FASTA or Gen GFA file containing DNA sequences")
    parser.add_argument("--format", choices=("fasta", "gfa"), help="Default: GFA for .gfa files, otherwise FASTA")
    parser.add_argument("--path", help="GFA path to score (required when the graph contains multiple paths)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Saved NVFlare checkpoint path")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}. Run job.py to train and save it first.")
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    input_format = args.format or ("gfa" if args.input.suffix.lower() == ".gfa" else "fasta")
    if args.path is not None and input_format != "gfa":
        parser.error("--path is only supported with GFA input")
    try:
        model = load_model(args.model)
        records = read_gfa(args.input, args.path) if input_format == "gfa" else read_fasta(args.input)
        for header, sequence in records:
            print(f"{predict(model, [sequence])[0]:.4f}\t{header}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
