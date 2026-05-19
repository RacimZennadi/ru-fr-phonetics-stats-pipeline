# Numerical Stability in Python

This project explores a simple but important idea: computers do not work with real numbers exactly. They work with finite approximations, and those approximations can change the outcome of numerical computations in subtle or sometimes dramatic ways.

The repository combines two parts:

- a didactic notebook showing classic numerical-stability issues in Python and NumPy;
- a reproducible DVC pipeline studying how reduced precision affects distances between speech embeddings extracted with wav2vec2.

The goal is not only to show that floating-point errors exist, but to study when they matter, how they appear, and whether they change the scientific conclusions we draw from data.

## Why This Project Exists

In machine learning and speech processing, most computations are done with floating-point numbers such as `float64`, `float32`, `float16`, or lower-precision quantised formats. Lower precision is attractive because it reduces memory usage and can speed up computation, but it also introduces approximation error.

This project asks two connected questions:

1. What kinds of numerical problems appear in ordinary Python computations?
2. When we reduce the precision of neural speech representations, do we merely perturb the numbers, or do we change the geometric structure of the representation space in a way that affects interpretation?

## Project Overview

The repository is organised around three deliverables:

- `numerical_stability.ipynb`
  An interactive notebook containing the theoretical answers and small-scale numerical experiments.
- `report.txt`
  A short written discussion of the final results and their interpretation.
- `dvc.yaml` + `src/`
  A reproducible pipeline for extracting, converting, analysing, and visualising wav2vec2 embeddings.

## What Is Studied in the Notebook

The notebook covers the conceptual and toy-experiment side of the project:

- why computers cannot represent all real numbers exactly;
- the role of the mantissa and exponent in floating-point numbers;
- rounding effects such as `0.1 + 0.2 != 0.3`;
- loss of associativity caused by finite precision;
- overflow and underflow in exponential functions;
- catastrophic cancellation in subtraction;
- the difference between a naive and numerically stable softmax implementation.

These examples build the intuition needed for the second part of the project, where numerical precision is studied in a realistic machine-learning setting.

## Speech-Embedding Experiment

The empirical part of the project uses repeated word productions from the Russian-French interference corpus. The idea is to extract one vector representation per word token using a pretrained wav2vec2 model, then compare how the same representations behave when stored in different numerical formats.

### Main idea

For each word token:

1. extract the corresponding audio segment from the corpus;
2. pass it through `facebook/wav2vec2-base`;
3. take the last hidden states;
4. average them over time to get a single word-level embedding;
5. save the embedding in `float64` as the reference version.

Then the same embedding matrix is converted into:

- `float32`
- `float16`
- `int8` using a simple min-max quantisation scheme

The project then compares the distance structure of the embedding space across these formats.

## Main Scientific Question

The key question is:

Does reducing precision only change the numerical values slightly, or does it fundamentally alter the structure of the representation space?

To answer this, the project computes cosine distances between word embeddings and compares:

- **intra-speaker distances**: same speaker, same word, different recordings;
- **inter-speaker distances**: different speakers, same word.

If reduced precision preserves the relative ordering and the broad separation between these two groups, then the structure of the space is considered stable for this analysis. If the distinction collapses or reverses, then reduced precision would be affecting the scientific interpretation.

## DVC Pipeline

The project uses DVC to keep the experiment organised and reproducible. The stages are defined in `dvc.yaml`.

### 1. `extract_embeddings`

Script: `src/extract_embeddings.py`

This stage:

- reads the corpus path and preprocessing parameters from `params.yaml`;
- loads the wav2vec2 processor and model;
- iterates through speaker folders and aligned word timestamp CSV files;
- extracts valid word segments;
- computes one mean-pooled embedding per token;
- saves:
  - `outputs/rep_float64.npy`
  - `outputs/labels.json`

The `float64` embeddings serve as the high-precision reference.

### 2. `convert_precision`

Script: `src/convert_precision.py`

This stage converts the reference embeddings into lower-precision versions:

- `outputs/rep_float32.npy`
- `outputs/rep_float16.npy`
- `outputs/rep_int8.npy`

It also saves the quantisation parameters in:

- `outputs/quant_params.json`

### 3. `compute_distances`

Script: `src/compute_distances.py`

This stage:

- computes cosine-distance matrices for each format;
- separates intra-speaker and inter-speaker comparisons;
- measures their averages and ratio;
- records timing information;
- saves:
  - `outputs/results.json`
  - `outputs/dist_matrices.npz`

### 4. `visualise`

Script: `src/visualise.py`

This stage turns the numerical results into figures:

- `figures/distance_distributions.png`
  Histograms of intra-speaker vs inter-speaker distances for each precision level.
- `figures/intra_inter_comparison.png`
  A compact comparison of the average distances across formats.

### 5. `analyse`

Script: `src/analyse.py`

This stage gathers summary metrics into:

- `outputs/analysis_summary.txt`

It reports:

- mean intra-speaker distance;
- mean inter-speaker distance;
- inter/intra ratio;
- computation time;
- file size;
- mean absolute error relative to the `float64` reference.

## Repository Structure

```text
.
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── numerical_stability.ipynb
├── report.txt
├── src/
│   ├── analyse.py
│   ├── compute_distances.py
│   ├── convert_precision.py
│   ├── extract_embeddings.py
│   └── visualise.py
├── outputs/
└── figures/
```

## Data and Parameters

The extraction stage expects the corpus to be available locally at the path defined in `params.yaml`:

`ru-fr_interference/ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only`

Current configurable parameters include:

- `corpus_root`
- `model_name`
- `num_speakers`
- `target_sr`
- `min_word_duration_s`

These make the pipeline easier to adapt without editing the source code directly.

## Installation

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Main libraries used:

- `numpy`
- `pandas`
- `matplotlib`
- `PyYAML`
- `torch`
- `torchaudio`
- `transformers`
- `dvc`

## Running the Pipeline

Once the corpus is in the expected location and the environment is installed, the full pipeline can be run with:

```bash
dvc repro
```

This will execute the stages in order and regenerate the saved outputs and figures.

## Interpreting the Results

The important point is not only whether the numbers change, but whether the interpretation changes.

Examples of outcomes:

- If file size drops a lot and the intra/inter-speaker structure stays nearly unchanged, then lower precision may be acceptable for this use case.
- If lower precision distorts the distance structure enough to blur or reverse the distinction between intra- and inter-speaker distances, then the scientific conclusion would no longer be trustworthy.

In this project, the observed behaviour suggests that lower precision mainly perturbed the values rather than destroying the structure, but that conclusion is specific to this setup and should not be generalised without caution.

## Limitations

Several limitations matter when reading the results:

- only a subset of speakers is used;
- embeddings are mean-pooled, which simplifies the temporal structure;
- the quantisation scheme is intentionally simple;
- the experiment is inference-only and CPU-based;
- different models, hardware, and tasks may show different numerical sensitivity.

So the project should be read as a careful case study in numerical stability, not as a universal claim that low precision is always safe.

## Takeaway

This repository is ultimately about numerical choices and their consequences.

At the toy level, numerical instability explains why mathematically equivalent expressions can behave differently on a computer. At the applied level, it shows that reducing precision can save memory and time, but must always be evaluated in relation to the question being asked. Sometimes lower precision only nudges the numbers; sometimes it changes the story.
