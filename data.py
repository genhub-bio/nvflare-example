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


def parse_gfa(path):
    """Read Gen GFA1 segments, paths, and links."""
    segments, paths, links = {}, {}, {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, start=1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            kind = fields[0]
            minimum = {"S": 3, "P": 4, "L": 6}.get(kind)
            if minimum is not None and len(fields) < minimum:
                raise ValueError(f"GFA line {number}: incomplete {kind} record")
            if kind == "S":
                if fields[1] in segments:
                    raise ValueError(f"GFA line {number}: duplicate segment {fields[1]!r}")
                segments[fields[1]] = fields[2]
            elif kind == "P":
                if fields[1] in paths:
                    raise ValueError(f"GFA line {number}: duplicate path {fields[1]!r}")
                paths[fields[1]] = (fields[2].split(","), fields[3])
            elif kind == "L":
                source, source_strand, target, target_strand, overlap = fields[1:6]
                if source_strand not in ("+", "-") or target_strand not in ("+", "-"):
                    raise ValueError(f"GFA line {number}: invalid link orientation")
                links[(source + source_strand, target + target_strand)] = overlap
                flip = {"+": "-", "-": "+"}
                links[(target + flip[target_strand], source + flip[source_strand])] = overlap
            elif kind not in ("H",):
                raise ValueError(f"GFA line {number}: unsupported {kind!r} record; expected Gen GFA1 S/L/P records")
    if not paths:
        raise ValueError("GFA contains no P paths")
    return segments, paths, links


def select_gfa_path(paths, path_name=None):
    if path_name is None:
        if len(paths) != 1:
            raise ValueError(
                "GFA contains multiple paths; select the child with --path. Available: " + ", ".join(paths)
            )
        path_name = next(iter(paths))
    if path_name not in paths:
        raise ValueError(f"GFA path {path_name!r} not found. Available: " + ", ".join(paths))
    return path_name


def gfa_sequence(graph, path_name):
    segments, paths, links = graph
    steps, overlaps = paths[path_name]
    if overlaps == "*":
        resolved_overlaps = [links.get(pair) for pair in zip(steps, steps[1:])]
    else:
        resolved_overlaps = overlaps.split(",")
        if len(resolved_overlaps) != len(steps) - 1:
            raise ValueError(f"GFA path {path_name!r}: incorrect overlap count")
    if any(overlap != "0M" for overlap in resolved_overlaps):
        raise ValueError(f"GFA path {path_name!r}: only explicit zero-overlap (0M) paths are supported")
    complement = str.maketrans("ACGTRYKMSWBDHVNacgtrykmswbdhvn", "TGCAYRMKSWVHDBNtgcayrmkswvhdbn")
    parts = []
    for step in steps:
        if len(step) < 2 or step[-1] not in ("+", "-"):
            raise ValueError(f"GFA path {path_name!r}: invalid oriented segment {step!r}")
        sequence = segments.get(step[:-1])
        if sequence is None or sequence == "*":
            raise ValueError(f"GFA path {path_name!r}: missing sequence for segment {step[:-1]!r}")
        parts.append(sequence if step[-1] == "+" else sequence.translate(complement)[::-1])
    return "".join(parts)


def read_gfa(path, path_name=None):
    """Read one named path from Gen's GFA1 output (explicit bases, zero overlaps)."""
    graph = parse_gfa(path)
    selected = select_gfa_path(graph[1], path_name)
    yield selected, gfa_sequence(graph, selected)
