# Numerical Stability in Python

This repository contains my submission for the numerical stability lab. It combines:

- a Jupyter notebook answering the theoretical questions and small numerical experiments;
- a DVC pipeline that extracts wav2vec2 word embeddings from the Russian-French interference corpus, converts them to lower-precision formats, computes cosine-distance statistics, and generates figures;
- a short report discussing whether reduced precision changes the scientific conclusions.

## Repository Structure

- `numerical_stability.ipynb`: theory answers, toy experiments, softmax case study, and interpretation of the embedding results.
- `report.txt`: one-page written discussion for the final question.
- `src/`: scripts used by the DVC pipeline.
- `dvc.yaml`: pipeline definition.
- `params.yaml`: configurable parameters for extraction and preprocessing.
- `outputs/` and `figures/`: generated artefacts from the last successful run.

## DVC Pipeline

The pipeline is organised into five stages:

1. `extract_embeddings`
2. `convert_precision`
3. `compute_distances`
4. `visualise`
5. `analyse`

The extraction stage expects the corpus to be available locally under the path configured in `params.yaml`:

`ru-fr_interference/ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only`

The pipeline was designed to be rerun with:

```bash
dvc repro
```

I did not rerun the stages during the final cleanup pass for submission, so the repository keeps the last generated artefacts.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `numpy`, `pandas`, `matplotlib`, `PyYAML`
- `torch`, `torchaudio`, `transformers`
- `dvc`

## Submission Contents

The intended submission is this Git repository, including:

- the notebook;
- the report;
- the DVC pipeline files;
- the source scripts;
- the generated figures and analysis outputs.
