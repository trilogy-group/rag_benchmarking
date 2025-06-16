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

The sample environment file leaves all variables blank. After copying it, create API keys for the providers you plan to use and replace the empty strings.

### Generating API Keys

- **OpenAI** – create a key at <https://platform.openai.com/account/api-keys>.
- **Azure** – create the required Azure resources (OpenAI and/or Search) in the [Azure Portal](https://portal.azure.com/) and copy the keys from the **Keys and Endpoint** page.
- **Pinecone** – sign in to <https://app.pinecone.io/> and generate an API key.
- **Qdrant** – sign up for <https://cloud.qdrant.io/> and create an API key.
- **Cohere** – go to <https://dashboard.cohere.ai> and generate a key.
- **Gemini** – obtain a key from <https://aistudio.google.com/app/apikey> (or through Vertex AI if using Google Cloud).
- **Vertex AI** – create a service account in Google Cloud and download a JSON credentials file, then set the `VERTEX_*` variables accordingly.
- **Milvus** – acquire your endpoint, username and password from your Milvus or Zilliz Cloud account.

Keep the `.env` file private and never commit it to version control.

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
