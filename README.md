# rag-benchmarking

This repository contains utilities for running retrieval augmented generation (RAG) experiments across various datasets and datastores.

## Installation

1. Clone the repository and install dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

2. Copy `env.sample` to `.env` and fill in the required API keys and settings:

```bash
cp env.sample .env
```

The application loads environment variables automatically when running experiments.

## Using the Interactive Shell

A simple interactive shell is provided to run common tasks. Start it with the `make` target:

```bash
make shell
```

You will be prompted to choose an action and an experiment. Available actions are:

- `download` – download only the dataset for the selected experiment
- `download_index` – download the dataset and index it into the configured datastore
- `evaluate` – run evaluation using the indexed dataset
- `full` – run download, index and evaluation in sequence
- `metrics` – consolidate metrics across previous runs
- `exit` – leave the shell

The shell maps each action to `src/main.py` which orchestrates the tasks.

## Running Experiments Manually

You can also invoke the same tasks directly without the shell:

```bash
# Example: run a full experiment
poetry run python src/main.py --task full --experiment QDRANT_FRAME_GEMINI_001_GPT_4O
```

Experiments are defined in `src/experiments.py` under the `ExperimentName` enum. Use any of these names with the `--experiment` flag.

Alternatively, the provided `make` targets mirror the available tasks:

```bash
make download EXP=<EXPERIMENT_NAME>
make download_index EXP=<EXPERIMENT_NAME>
make evaluate EXP=<EXPERIMENT_NAME>
make full EXP=<EXPERIMENT_NAME>
```

## Adding More Dependencies

This project manages dependencies with Poetry. To add a new package:

1. Run `poetry add <package>` which updates `pyproject.toml` and `poetry.lock`.
2. Commit both updated files so that others install the same versions.

For development-only dependencies, use `poetry add --group dev <package>`.

*** End of File
