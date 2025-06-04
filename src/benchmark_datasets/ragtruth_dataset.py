from benchmark_datasets._benchmark_dataset import BenchmarkDataset, BenchmarkData
from datasets import load_dataset
from typing import Dict

class RagTruthDataset(BenchmarkDataset):
    """Dataset loader for the RAGTruth dataset hosted on HuggingFace."""

    def __init__(self, hf_dataset: str = "Azure/ragtruth", subset: str = "default", split: str = "test"):
        self.hf_dataset = hf_dataset
        self.subset = subset
        self.split = split
        self.queries: Dict[str, str] = {}
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.relevant_docs: Dict[str, list] = {}

    def load(self) -> BenchmarkData:
        ds = load_dataset(self.hf_dataset, self.subset, split=self.split)
        for idx, row in enumerate(ds):
            doc_id = str(idx)
            self.corpus[doc_id] = {"text": row.get("document", row.get("context", ""))}
            self.queries[doc_id] = row.get("query", row.get("question", ""))
            self.relevant_docs[doc_id] = [doc_id]
        return BenchmarkData(corpus=self.corpus, queries=self.queries, relevant_docs=self.relevant_docs)

    def get_corpus(self) -> Dict[str, Dict[str, str]]:
        return self.corpus

    def get_queries(self) -> Dict[str, str]:
        return self.queries

    def get_relevant_docs(self) -> Dict[str, list]:
        return self.relevant_docs
