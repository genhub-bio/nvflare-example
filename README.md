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
`organization-b`. Before training, Python clones
`https://staging.genhub.bio/api/repos/organization-a/nvflare-data` into `organization-a` and
`https://staging.genhub.bio/api/repos/organization-b/nvflare-data-b` into `organization-b` using `gen.clone`.
Existing local Gen repositories are reused. Both local repository directories are deleted when the run ends, including on failure. Use `--org_a_repo` and `--org_b_repo` to choose other local paths; these directories are also deleted.

## Benchmarking

To reproduce the CPU benchmark explicitly:

```bash
python job.py --num_samples 160 --eval_samples 40 --num_rounds 1 --device cpu
```

Each client prints loading/encoding time, evaluation time, training time per
round, and average training time per genome. The launcher also prints federated
wall time, including startup, communication, aggregation, and final evaluation.
CUDA/MPS work is synchronized before recording times. The automatic device
selection prefers CUDA, then Apple MPS, then CPU. Each client uses at most four CPU threads.

Sample counts are **per client** and must fit within that client's training and
evaluation splits. Each supplied organization has 200 scored samples: the
reproducible split puts 160 in training and 40 in evaluation. The benchmark limits
select from those separate splits before loading DNA, so no training genome is
used for evaluation. Five samples are for timing, not a reliable quality estimate.

## Complete sequences and model

The client reads complete DNA sequences from each organization's Gen repository and uses sample metadata (`Score`
by default) as the regression target. Unscored samples are excluded, and scored samples are split reproducibly into
training and held-out evaluation sets. Every A/C/G/T base in each selected genome is encoded and used.

The CNN learns local sequence patterns with two convolutional layers, then averages features across 100 ordered
regions to preserve coarse position within the genome before predicting a score. Overlapping chunks and activation
checkpointing keep memory use manageable while preserving context across chunk boundaries. Training uses one complete
genome per optimizer step and one local epoch per federated round.

Each client normalizes targets using its training scores and converts predictions back to original score units for
evaluation and FedAvg aggregation. Clients report mean squared error (MSE) on held-out genomes alongside a baseline
that always predicts their training-score mean; lower MSE is better. Training-set metrics help assess local fitting,
and a final evaluation measures the aggregated model on each client's training and held-out samples.

## Prediction

A successful run saves `models/global_model.pt`, including the CNN weights and architecture configuration. It replaces
the previous checkpoint. Earlier checkpoints (v1/v2/v3) are incompatible; rerun `job.py` to
produce a strided regional-pooling CNN (v4) model. The checkpoint records the channel and region counts used for prediction.


## Files

- `job.py`: federated recipe, per-client settings, timing, and checkpoint export.
- `client.py`: full-sequence datasets, local training, evaluation, and timings.
- `data.py`: Gen sample selection and validation.
- `model.py`: shared CNN, full-sequence encoding, and device helpers.
- `predict.py`: score FASTA sequences with the saved CNN.
- `benchmark_full_sequences.py`: local timing using the same model.
- `organization-a/`, `organization-b/`: local Gen repositories and source data.
- `models/global_model.pt`: generated federated CNN checkpoint.
