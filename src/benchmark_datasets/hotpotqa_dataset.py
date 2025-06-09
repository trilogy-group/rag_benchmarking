from benchmark_datasets._benchmark_dataset import BenchmarkDataset, BenchmarkData
from typing import List, Dict
import json
import os
import pathlib
from datasets import load_dataset, load_from_disk
from collections import defaultdict
import hashlib

class HotpotQADataset(BenchmarkDataset):
    def __init__(self, dataset_name: str, base_path: str = "./data/hotpotqa", split="validation"):
        self.base_path = pathlib.Path(base_path)
        self.dataset_path = self.base_path / split
        self.split = split
        self.queries = {}
        self.answers = {}
        self.corpus = {}
        self.relevant_docs = {}

    def load(self) -> BenchmarkData:
        if self.dataset_path.exists():
            print(f"📁 Loading HotpotQA dataset from local cache: {self.dataset_path}")
            data = load_from_disk(str(self.dataset_path))
        else:
            print("🌐 Downloading HotpotQA dataset from Hugging Face...")
            data = load_dataset("hotpot_qa", "distractor", split=self.split, trust_remote_code=True)
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            data.save_to_disk(str(self.dataset_path))
            print(f"✅ Saved HotpotQA dataset to: {self.dataset_path}")

        qrels = defaultdict(dict)
        output_dir = self.base_path / "retrieval_format"
        output_dir.mkdir(parents=True, exist_ok=True)

        queries_file = output_dir / "queries.json"
        corpus_file = output_dir / "corpus.json"
        qrels_file = output_dir / "qrels.json"

        try:
            with open(queries_file, "r") as f:
                queries_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            queries_data = {}

        try:
            with open(corpus_file, "r") as f:
                corpus_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            corpus_data = {}

        try:
            with open(qrels_file, "r") as f:
                qrels_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            qrels_data = {}

        seen_doc_ids = set(corpus_data.keys())
        self.corpus.update(corpus_data)

        print(f"Number of items in dataset: {len(data)}")

        if len(qrels_data) < 50:
            for i, item in enumerate(data):
                qid = f"q{i}"
                if qid not in qrels_data:
                    print(f"\n🔍 Query {i}/{len(data)}: {item['question']}")
                    self.queries[qid] = item["question"]
                    self.answers[qid] = item["answer"]
                    queries_data[qid] = item["question"]

                    support_titles = set(item["supporting_facts"]["title"])

                    for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                        paragraph = " ".join(sentences)
                        doc_id = hashlib.md5((title + paragraph).encode('utf-8')).hexdigest()

                        if doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            doc_entry = {
                                "title": title,
                                "text": paragraph
                            }
                            self.corpus[doc_id] = doc_entry
                            corpus_data[doc_id] = doc_entry

                        if title in support_titles:
                            qrels[qid][doc_id] = 1
                            qrels_data[qid] = qrels[qid]

                    with open(queries_file, "w") as f:
                        json.dump(queries_data, f, indent=2)
                    with open(corpus_file, "w") as f:
                        json.dump(corpus_data, f, indent=2)
                    with open(qrels_file, "w") as f:
                        json.dump(qrels_data, f, indent=2)
            print(f"✅ Converted {len(self.queries)} queries and {len(self.corpus)} corpus documents")

        print(f"📁 Saved retrieval format to {output_dir}")

        self.queries = queries_data
        self.corpus = corpus_data
        self.relevant_docs = qrels_data

        return BenchmarkData(
            corpus=self.corpus,
            queries=self.queries,
            relevant_docs=self.relevant_docs
        )
