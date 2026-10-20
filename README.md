# Multiple organizations: full-sequence score regression

In this repo, two organizations can train a shared CNN using complete DNA sequences from their local gen
repositories. Each genome has one target score from sample metadata.  Sequences and scores remain on clients; NVFlare
aggregates model parameters.

## Run federated training with the full data split

Use Python 3.11 or newer with NVFlare installed. From this directory:

```bash
python -m pip install -r requirements.txt
python job.py
```

The defaults run **160 training genomes and 40 held-out evaluation genomes per client** (320 training and 80 evaluation
genomes total), one local epoch, and one federated round. `site-1` uses `organization-a`, and `site-2` uses
`organization-b`. Each must contain an existing gen repository with its sequence assets.

## Benchmarking

To reproduce the CPU benchmark explicitly:

```bash
python job.py --num_samples 160 --eval_samples 40 --epochs 1 --num_rounds 1 --device cpu
```

Each client prints loading/encoding time, evaluation time, training time per
round, and average training time per genome. The launcher also prints federated
wall time, including startup, communication, aggregation, and final evaluation.
CUDA/MPS work is synchronized before recording times. The automatic device
selection prefers CUDA, then Apple MPS, then CPU. `--threads` defaults to at most
four CPU threads per client.

Sample counts are **per client** and must fit within that client's training and
evaluation splits. Each supplied organization has 200 scored samples: the
reproducible split puts 160 in training and 40 in evaluation. The benchmark limits
select from those separate splits before loading DNA, so no training genome is
used for evaluation. Five samples are for timing, not a reliable quality estimate.

## Complete sequences and model

The client uses the gen python package to get information from the gen repositories.  It lists samples through
`repo.samples`, selects those with `Score` metadata, and reads the complete sequence from each sample's sequence graph.
We exclude the unscored reference. We sort samples by collection and name, then shuffle them with a seed of 42 before
the held-out split. Each selected sample must have exactly one graph and one sequence, with a finite float32 score. The
client accepts `--score_key` to use another metadata field.

Every A/C/G/T base is encoded and used. Training generates no GFA files, alignments, reference-relative variants, or
k-mer vectors. The CNN has two stride-one convolutional layers (kernel sizes 9 and 5, default 8 channels), ReLU
activations, global average pooling, and a regression output. There is one loss and one optimizer step per complete
genome, with batch size 1.

Overlapping chunks preserve convolution context at their boundaries. Activation checkpointing recomputes chunk
activations during backward to bound memory; chunk outputs are combined before computing the genome's score. The default
chunk size is 65,536 bases. Increase model width or change memory usage with:

```bash
python job.py --channels 32 --chunk_size 131072
```

The default model has 633 parameters. Its cost includes processing every base, even though its checkpoint is small. This
is an example CNN: global pooling loses absolute positions and the model has a short local receptive field.  Timing
measures this architecture, not a large genomic transformer.

Clients report mean squared error (MSE) on the incoming global model, then train locally and send parameters for FedAvg
aggregation. Lower MSE is better. A final evaluation measures the aggregated model on each client's held-out genomes.

## Prediction

A successful run saves `models/global_model.pt`, including the CNN weights and architecture configuration. It replaces
the previous checkpoint. Earlier k-mer and variant checkpoints are incompatible; rerun `job.py` to produce a CNN model.


## Files

- `job.py`: federated recipe, per-client settings, timing, and checkpoint export.
- `client.py`: full-sequence datasets, local training, evaluation, and timings.
- `data.py`: Gen sample selection, validation, and FASTA/GFA readers.
- `model.py`: shared CNN, full-sequence encoding, and device helpers.
- `predict.py`: score FASTA or GFA sequences with the saved CNN.
- `benchmark_full_sequences.py`: local timing using the same model.
- `organization-a/`, `organization-b/`: local Gen repositories and source data.
- `models/global_model.pt`: generated federated CNN checkpoint.
