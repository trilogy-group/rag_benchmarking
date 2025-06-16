from benchmark_datasets._benchmark_dataset import BenchmarkDataset, BenchmarkData
from typing import List, Dict
import json
import os
import pathlib
import urllib.request
from datasets import load_dataset, load_from_disk, Dataset
from collections import defaultdict
import requests
from urllib.parse import urlparse, unquote
import hashlib
import logging

logger = logging.getLogger(__name__)

class FrameDataset(BenchmarkDataset):
    def __init__(self, dataset_name: str, base_path:str="./data/benchmark_datasets/frames", split="test"):
        self.base_path = pathlib.Path(base_path)
        self.dataset_name = dataset_name
        self.dataset_path = self.base_path / dataset_name
        self.split = split
        self.queries = {}
        self.answers = {}
        self.evidence = {}
        self.corpus = {}
        self.relevant_docs = {}
    
    def extract_title_from_url(self, wiki_url: str) -> str:
        parsed = urlparse(wiki_url)
        return unquote(parsed.path.split("/wiki/")[-1])

    def load(self) -> BenchmarkData:
        if self.dataset_path.exists():
            logger.info(f"Loading FRAME dataset from local cache: {self.dataset_path}")
            data = load_from_disk(str(self.dataset_path))
        else:
            logger.info("Downloading FRAME dataset from Hugging Face...")
            data = load_dataset("google/frames-benchmark", split=self.split)
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            data.save_to_disk(str(self.dataset_path))
            logger.info(f"Saved FRAME dataset to: {self.dataset_path}")

        qrels = defaultdict(dict)
        output_dir = self.base_path / "retrieval_format"
        output_dir.mkdir(parents=True, exist_ok=True)

        queries_file = output_dir / "queries.json"
        corpus_file = output_dir / "corpus.json"
        qrels_file = output_dir / "qrels.json"

        # Load existing data if present
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

        logger.info(f"Number of items in dataset: {len(data)}")

        # The dataset has 824 queries
        if len(qrels_data) < 50:
            for i, item in enumerate(data):
                qid = f"q{i}"
                if qid not in qrels_data:
                    logger.debug(f"Query {i}/{len(data)}: {item['Prompt']}")
                    self.queries[qid] = item["Prompt"]
                    self.answers[qid] = item["Answer"]
                    queries_data[qid] = item["Prompt"]

                    links = [item.get(f"wikipedia_link_{j}") for j in range(1, 12)]
                    links = [l.strip() for l in links if l and isinstance(l, str) and l.strip()]

                    for link in links:
                        title = self.extract_title_from_url(link)
                        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                        logger.debug(f"Fetching: {url}")
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                response = requests.get(url, timeout=10)
                                if response.status_code == 200:
                                    doc = response.json()
                                    doc_id = hashlib.md5(doc["title"].encode('utf-8')).hexdigest()

                                    if doc_id not in seen_doc_ids:
                                        seen_doc_ids.add(doc_id)
                                        doc_entry = {
                                            "title": doc.get("title", ""),
                                            "text": doc.get("extract", "")
                                        }
                                        self.corpus[doc_id] = doc_entry
                                        corpus_data[doc_id] = doc_entry

                                    qrels[qid][doc_id] = 1
                                    qrels_data[qid] = qrels[qid]
                                    break  # Exit the retry loop on success
                                else:
                                    logger.warning(f"Failed with status {response.status_code}: {url}")
                            except Exception as e:
                                logger.warning(f"Attempt {attempt + 1} failed to fetch {link}: {e}")
                                if attempt == max_retries - 1:
                                    logger.error(f"All retry attempts failed for {link}")
                                logger.error(f"Failed to fetch {link}: {e}")

                    # Write updated state
                    with open(queries_file, "w") as f:
                        json.dump(queries_data, f, indent=2)
                    with open(corpus_file, "w") as f:
                        json.dump(corpus_data, f, indent=2)
                    with open(qrels_file, "w") as f:
                        json.dump(qrels_data, f, indent=2)
            logger.info(f"Converted {len(self.queries)} queries and {len(self.corpus)} corpus documents")
        logger.info(f"Saved retrieval format to {output_dir}")

        self.queries = queries_data
        self.corpus = corpus_data
        self.relevant_docs = qrels_data

        return BenchmarkData(
            corpus=self.corpus,
            queries=self.queries,
            relevant_docs=self.relevant_docs
        )
    
