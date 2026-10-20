"""Select scored Gen samples and read complete DNA sequences."""

import math
import random
from pathlib import Path


def open_repository(path):
    """Open an existing repository; do not silently initialize a missing one."""
    import gen

    path = Path(path).expanduser().resolve()
    if not (path / ".gen" / "gen.db").is_file():
        raise ValueError(f"Not an existing Gen repository: {path}")
    return gen.Repository(str(path))


def split_samples(repo, score_key="Score", seed=42):
    """Select labeled samples, then reserve 20% for evaluation before reading DNA."""
    samples = sorted((s for s in repo.samples if score_key in s.metadata), key=lambda s: (s.collection, s.name))
    if len(samples) < 2:
        raise ValueError(f"Repository must contain at least two samples with {score_key!r} metadata")
    random.Random(seed).shuffle(samples)
    evaluation_size = max(1, len(samples) // 5)
    return samples[evaluation_size:], samples[:evaluation_size]


def sample_score(sample, score_key="Score"):
    try:
        score = float(sample.metadata[score_key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Sample {sample.name}: {score_key} must be numeric") from error
    if not math.isfinite(score) or abs(score) > 3.4028234663852886e38:
        raise ValueError(f"Sample {sample.name}: {score_key} must be a finite float32 value")
    return score


def sample_rows(samples, score_key="Score"):
    """Yield one (sequence, score) per sample, keeping only one genome in memory."""
    for sample in samples:
        score = sample_score(sample, score_key)
        graphs = sample.sequence_graphs
        if len(graphs) != 1:
            raise ValueError(f"Sample {sample.name}: expected exactly one sequence graph")
        sequences = graphs[0].all_sequences()
        if len(sequences) != 1:
            raise ValueError(f"Sample {sample.name}: expected exactly one sequence")
        sequence = str(sequences[0])
        if not sequence:
            raise ValueError(f"Sample {sample.name}: sequence must not be empty")
        yield sequence, score


def dna_sequence(sequence):
    sequence = sequence.upper()
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("Expected a nonempty DNA sequence containing only A/C/G/T")
    return sequence


def read_fasta(path):
    """Yield (header, sequence) records, joining wrapped sequence lines."""
    header = None
    parts = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    if not parts:
                        raise ValueError(f"FASTA record {header!r} has no sequence")
                    yield header, "".join(parts)
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"FASTA line {number}: header must not be empty")
                parts = []
            else:
                if header is None:
                    raise ValueError(f"FASTA line {number}: expected a '>' header before sequence data")
                parts.append("".join(line.split()))
    if header is None:
        raise ValueError("FASTA file contains no records")
    if not parts:
        raise ValueError(f"FASTA record {header!r} has no sequence")
    yield header, "".join(parts)
