from benchmark_datasets._benchmark_dataset import BenchmarkDataset, BenchmarkData
from datasets import load_dataset, load_from_disk
from typing import Dict
import pathlib
import os


class RagTruthDataset(BenchmarkDataset):
    """Dataset loader for the RAGTruth dataset hosted on HuggingFace, with local caching."""

    def __init__(self, dataset_name: str = "default", base_path: str = "./data/ragtruth", split: str = "test"):
        self.hf_dataset = "wandb/RAGTruth-processed"
        self.subset = dataset_name
        self.split = split
        self.base_path = pathlib.Path(base_path)
        self.dataset_path = self.base_path / dataset_name
        self.queries: Dict[str, str] = {}
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.relevant_docs: Dict[str, list] = {}

    def load(self) -> BenchmarkData:
        if self.dataset_path.exists():
            print(f"📁 Loading RAGTruth dataset from local cache: {self.dataset_path}")
            ds = load_from_disk(str(self.dataset_path))
        else:
            print("🌐 Downloading RAGTruth dataset from Hugging Face...")
            ds = load_dataset(self.hf_dataset, self.subset, split=self.split)
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(self.dataset_path))
            print(f"✅ Saved RAGTruth dataset to: {self.dataset_path}")

        print(f"📊 Number of items in dataset: {len(ds)}")
        for idx, row in enumerate(ds):
            doc_id = str(idx)
            self.corpus[doc_id] = {"text": row.get("document", row.get("context", ""))}
            self.queries[doc_id] = row.get("query", row.get("question", ""))
            self.relevant_docs[doc_id] = [doc_id]

        print(f"✅ Loaded {len(self.queries)} queries and {len(self.corpus)} documents")
        return BenchmarkData(
            corpus=self.corpus,
            queries=self.queries,
            relevant_docs=self.relevant_docs
        )
