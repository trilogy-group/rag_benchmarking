# rag-benchmarking

This repository provides simple utilities for benchmarking retrieval systems with Azure AI Search.

## Datasets
- **BEIR** – loads datasets using `BeirDataset`.
- **RAGTruth** – loads the [RAGTruth](https://github.com/microsoft/RAGTruth) dataset via HuggingFace using `RagTruthDataset`.

## Running an Experiment
Experiments can be configured in `src/main.py`. Example for RAGTruth:

```python
from benchmark_datasets.ragtruth_dataset import RagTruthDataset
from datastores.azure_ai_search_store import AzureAISearchStore
from retrievers.azure_ai_search_retriever import AzureAISearchRetriever
from evaluators.beir_evaluator import BEIREvaluator

# ... set up experiment runner
```

Use `python -m pip install -r requirements.txt` (or `poetry install`) to install dependencies including `datasets` for RAGTruth support.
