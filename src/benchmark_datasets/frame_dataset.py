from benchmark_datasets._benchmark_dataset import BenchmarkDataset, BenchmarkData
from typing import List, Dict
import json
import os
import pathlib
import urllib.request
from datasets import load_dataset, load_from_disk, Dataset
from collections import defaultdict
import requests

class FrameDataset(BenchmarkDataset):
    def __init__(self, dataset_name: str, base_path:str="./data/frames", split="test"):
        self.base_path = pathlib.Path(base_path)
        self.dataset_name = dataset_name
        self.dataset_path = self.base_path / dataset_name
        self.split = split
        self.queries = {}
        self.answers = {}
        self.evidence = {}
        self.corpus = {}
    
    def load(self) -> BenchmarkData:
        if self.dataset_path.exists():
            print(f"📁 Loading FRAME dataset from local cache: {self.dataset_path}")
            data = load_from_disk(str(self.dataset_path))
        else:
            print("🌐 Downloading FRAME dataset from Hugging Face...")
            data = load_dataset("google/frames-benchmark", split=self.split)
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            data.save_to_disk(str(self.dataset_path))
            print(f"✅ Saved FRAME dataset to: {self.dataset_path}")

        qrels = defaultdict(dict)
        print(f"Number of items in dataset: {len(data)}")
        for i, item in enumerate(data):
            print(f"Processing item index: {i} of {len(data)}")
            qid = f"q{i}"
            self.queries[qid] = item["Prompt"]
            self.answers[qid] = item["Answer"]

            # Parse gold links
            links = [item.get(f"wikipedia_link_{j}") for j in range(1, 12)]
            links = [l.strip() for l in links if l and isinstance(l, str) and l.strip()]

            for l in links:
                try:
                    print(f"Fetching from wiki {l}")
                    response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{l.replace(' ', '_')}")
                    if response.status_code == 200:
                        doc = response.json()
                        doc_id = doc["title"].replace(" ", "_")
                        self.corpus[doc_id] = {
                            "title": doc.get("title", ""),
                            "text": doc.get("extract", "")
                        }
                        qrels[qid][doc_id] = 1
                except Exception as e:
                    print(f"❌ Failed to fetch {l}: {e}")

        print(f"✅ Converted {len(self.queries)} queries and {len(self.corpus)} corpus documents")

        # Save to disk for reuse
        output_dir = self.base_path / "retrieval_format"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "queries.json", "w") as f:
            json.dump(self.queries, f, indent=2)
        with open(output_dir / "corpus.json", "w") as f:
            json.dump(self.corpus, f, indent=2)
        with open(output_dir / "qrels.json", "w") as f:
            json.dump(qrels, f, indent=2)

        print(f"📁 Saved retrieval format to {output_dir}")

        return BenchmarkData(
            corpus=self.corpus,
            queries=self.queries,
            relevant_docs=qrels
        )

    # def load(self) -> BenchmarkData:
    #     if self.dataset_path.exists():
    #         print(f"📁 Loading FRAME dataset from local cache: {self.dataset_path}")
    #         data = load_from_disk(str(self.dataset_path))
    #     else:
    #         print("🌐 Downloading FRAME dataset from Hugging Face...")
    #         data = load_dataset("google/frames-benchmark", split=self.split)
    #         self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    #         data.save_to_disk(str(self.dataset_path))
    #         print(f"✅ Saved FRAME dataset to: {self.dataset_path}")

    #     # Load the KILT Wikipedia dataset
    #     print("🌐 Loading KILT Wikipedia corpus...")
    #     wiki = load_dataset("kilt_wikipedia", split="full", trust_remote_code=True)
    #     wiki_docs = {str(row['wikipedia_id']): {"title": row["title"], "text": row["text"]} for row in wiki}
    #     print(f"✅ Loaded {len(wiki_docs)} Wikipedia documents.")

    #     qrels = defaultdict(dict)
    #     for i, item in enumerate(data):
    #         qid = f"q{i}"
    #         self.queries[qid] = item["Prompt"]
    #         self.answers[qid] = item["Answer"]

    #         # Parse gold links
    #         links = [item.get(f"wikipedia_link_{j}") for j in range(1, 12)]
    #         links = [l.strip() for l in links if l and isinstance(l, str) and l.strip()]

    #         for l in links:
    #             # match article by title (note: KILT uses titles)
    #             matching_ids = [k for k, doc in wiki_docs.items() if doc["title"] == l]
    #             if not matching_ids:
    #                 continue
    #             for doc_id in matching_ids:
    #                 self.corpus[doc_id] = wiki_docs[doc_id]
    #                 qrels[qid][doc_id] = 1

    #     print(f"✅ Converted {len(self.queries)} queries and {len(self.corpus)} corpus documents")

    #     # Save to disk for reuse
    #     output_dir = self.base_path / "retrieval_format"
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     with open(output_dir / "queries.json", "w") as f:
    #         json.dump(self.queries, f, indent=2)
    #     with open(output_dir / "corpus.json", "w") as f:
    #         json.dump(self.corpus, f, indent=2)
    #     with open(output_dir / "qrels.json", "w") as f:
    #         json.dump(qrels, f, indent=2)

    #     print(f"📁 Saved retrieval format to {output_dir}")

    #     return BenchmarkData(
    #         corpus=self.corpus,
    #         queries=self.queries,
    #         relevant_docs=qrels
    #     )
