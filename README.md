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

The client uses the gen python package to get information from the gen repositories.  It lists samples through
`repo.samples`, selects those with `Score` metadata, and reads the complete sequence from each sample's sequence graph.
We exclude the unscored reference. We sort samples by collection and name, then shuffle them with a seed of 42 before
the held-out split. Each selected sample must have exactly one graph and one sequence, with a finite float32 score. The
client accepts `--score_key` to use another metadata field.

Every A/C/G/T base is encoded and used. Training generates no GFA files, alignments, reference-relative variants, or
k-mer vectors. The CNN has 10 channels, a first convolution with a 100-base kernel and stride 100, and a second convolution with
kernel size 5 and stride 1. LeakyReLU activations are followed by average pooling within 100 ordered regions and
a regression output. There is one loss and one optimizer step per complete
genome, with batch size 1.

Overlapping chunks preserve convolution context at their boundaries. Activation checkpointing recomputes chunk
activations during backward to bound memory; chunk outputs are combined before computing the genome's score. The
chunk size is 65,536 bases (rounded down to 655 whole windows), with 10 channels and one local training epoch per round.
The first convolution processes nonoverlapping 100-base windows; an incomplete final window is zero-padded. The second
convolution combines five neighboring windows, giving a 500-base receptive field and a total stride of 100 bases.
Chunks include two context windows on either side and always align to the same genome-wide window grid.

The output feature positions are divided into 100 consecutive regions of approximately equal length. Each region contributes 10 average
features, giving the output layer 1,000 inputs. This preserves coarse relative position: the same pattern can have a
different effect in different regions. Context crosses region boundaries, and every output position contributes to exactly one
regional average. When there are fewer output positions than regions, empty regions contribute zeros. For a
4,641,652-base genome there are 46,417 output positions, so each region pools 464 or 465 positions.

The default model has 5,521 parameters. It still has a short local receptive field and discards positions within each
region. More features or regions do not guarantee fitting random scores or improving held-out accuracy.

Clients report mean squared error (MSE) on the incoming global model, then train locally and send parameters for FedAvg
aggregation. Lower MSE is better. A final evaluation measures the aggregated model on each client's held-out genomes.

Each client standardizes targets using the mean and population standard deviation of its selected training scores
only (a scale of 1 is used for constant targets). The final layer is converted into these local units before training
and back to original score units before evaluation and federated aggregation. Normalization statistics stay local;
saved models predict in original score units. LeakyReLU keeps gradients flowing through negative activations.

Evaluation logs include a baseline MSE from always predicting that client's training-score mean. Positive
`improvement` means the model beats this baseline on held-out data. The online training MSE summarizes predictions
made during updates. A separate training evaluation after each local round reports MSE with fixed weights, the
training-mean baseline MSE, and prediction standard deviation and range. The final aggregated model is also evaluated
on each client's training samples, so you can distinguish local fitting from the result of aggregation. These passes
add evaluation time but do not update weights or change the training shuffle order. A narrow prediction range and
training MSE close to the baseline indicate that the model is mostly predicting a constant.

To compare with the earlier 100-region experiment, retrain with:

```bash
python job.py --num_rounds 10
```

New runs use 100 regions; loading an existing v4 checkpoint uses its recorded region count.

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
