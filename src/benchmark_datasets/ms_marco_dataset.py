from benchmark_datasets._benchmark_dataset import BenchmarkDataset, BenchmarkData
from typing import Dict
import json
import os
import pathlib
from datasets import load_dataset, load_from_disk
from collections import defaultdict

class MSMarcoDataset(BenchmarkDataset):
    def __init__(self, dataset_name: str = "msmarco", base_path: str = "./data/benchmark_datasets/msmarco", split="validation", max_queries: int = 1000):
        self.dataset_name = dataset_name
        self.base_path = pathlib.Path(base_path)
        self.dataset_path = self.base_path / split
        self.split = split
        self.max_queries = max_queries  # limit for subsampling
        self.queries = {}
        self.answers = {}
        self.corpus = {}
        self.relevant_docs = {}

    def load(self) -> BenchmarkData:
        if self.dataset_path.exists():
            print(f"📁 Loading MS MARCO dataset from local cache: {self.dataset_path}")
            data = load_from_disk(str(self.dataset_path))
        else:
            print("🌐 Downloading MS MARCO dataset from Hugging Face...")
            data = load_dataset("ms_marco", "v1.1", split=self.split)
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            data.save_to_disk(str(self.dataset_path))
            print(f"✅ Saved MS MARCO dataset to: {self.dataset_path}")

        output_dir = self.base_path / "retrieval_format"
        output_dir.mkdir(parents=True, exist_ok=True)

        queries_file = output_dir / "queries.json"
        corpus_file = output_dir / "corpus.json"
        qrels_file = output_dir / "qrels.json"

        # Load pre-existing data if available
        self.queries = self._load_json(queries_file)
        self.corpus = self._load_json(corpus_file)
        self.relevant_docs = self._load_json(qrels_file)

        if len(self.relevant_docs) < 50:
            print(f"🔄 Converting up to {self.max_queries} entries into BEIR format")
            qrels = defaultdict(dict)
            seen_doc_ids = set(self.corpus.keys())

            for i, item in enumerate(data):
                if i >= self.max_queries:
                    break

                qid = f"q{i}"
                query = item["query"]
                self.queries[qid] = query

                passages = item["passages"]["passage_text"]
                labels = item["passages"]["is_selected"]

                for j, (text, label) in enumerate(zip(passages, labels)):
                    doc_id = f"{qid}_d{j}"
                    if doc_id not in seen_doc_ids:
                        self.corpus[doc_id] = {"title": "", "text": text}
                        seen_doc_ids.add(doc_id)
                    if label == 1:
                        qrels[qid][doc_id] = 1

                if i % 100 == 0:
                    print(f"Processed {i}/{self.max_queries} queries...")

            self.relevant_docs = dict(qrels)

            self._save_json(self.queries, queries_file)
            self._save_json(self.corpus, corpus_file)
            self._save_json(self.relevant_docs, qrels_file)

            print(f"✅ Final: {len(self.queries)} queries | {len(self.corpus)} documents")

        return BenchmarkData(
            corpus=self.corpus,
            queries=self.queries,
            relevant_docs=self.relevant_docs
        )

    def _load_json(self, path: pathlib.Path) -> Dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_json(self, obj: Dict, path: pathlib.Path):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)