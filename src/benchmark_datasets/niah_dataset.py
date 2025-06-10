from benchmark_datasets._benchmark_dataset import BenchmarkDataset, BenchmarkData
from typing import Dict, List
import json
import os
import pathlib
from datasets import load_dataset, load_from_disk
import requests
import zipfile
import io
import tiktoken
from tqdm import tqdm
            

class NiahDataset(BenchmarkDataset):
    def __init__(self, dataset_name: str = "niah", base_path: str = "./data/benchmark_datasets/niah", split="validation", max_queries: int = 1000):
        self.dataset_name = dataset_name
        self.base_path = pathlib.Path(base_path)
        self.dataset_path = self.base_path / split
        self.split = split
        self.max_queries = max_queries  # limit for subsampling
        self.queries = {}
        self.answers = {}
        self.corpus = {}
        self.relevant_docs = {}
        self.output_json_path = self.base_path / "trilogy_niah_essay_repo.json"
        self.dataset_path = self.base_path


    def download_and_unzip(self, url: str, output_dir: pathlib.Path):
        """Download and unzip a dataset from a URL to the specified output directory.
        Only downloads if the target directory doesn't exist or is empty.
        
        Args:
            url (str): URL of the zip file to download
            output_dir (pathlib.Path): Directory to extract the contents to
        """
        
        # Check if directory exists and has contents
        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"Dataset already exists at {output_dir}, skipping download.")
            return
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the zip file
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the zip file
        print(f"Extracting to {output_dir}...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(output_dir)
        print("Download and extraction complete.")


    def create_sorted_haystack(self, haystack_files: List[pathlib.Path], output_file: pathlib.Path) -> None:
        """Create a sorted haystack JSONL file from multiple input files.
        
        Args:
            haystack_files (List[pathlib.Path]): List of input JSONL files
            output_file (pathlib.Path): Path to write the sorted output file
        """
        # Initialize tokenizer
        enc = tiktoken.get_encoding("cl100k_base")
        
        # Collect all items with their token counts
        all_items = []
        seen_texts = set()  # Track unique texts
        
        for file_path in haystack_files:
            if not file_path.exists():
                print(f"Warning: File {file_path} does not exist, skipping...")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = obj.get('text', '')
                        
                        # Skip if we've seen this text before
                        if text in seen_texts:
                            continue
                            
                        seen_texts.add(text)
                        # Calculate token count
                        token_count = len(enc.encode(text))
                        obj['token_count'] = token_count
                        all_items.append(obj)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse JSON from {file_path}")
                        continue

        # Sort items by token count
        all_items.sort(key=lambda x: x['token_count'])

        # Write to output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Created haystack file with {len(all_items)} unique items at {output_file}")

    def insert_needles_into_haystack(self, needles_file: pathlib.Path, haystack_file: pathlib.Path, output_file: pathlib.Path) -> None:
        """Insert needles into haystack at different depths.
        
        Args:
            needles_file (pathlib.Path): Path to the needles.jsonl file
            haystack_file (pathlib.Path): Path to the sorted haystack file
            output_file (pathlib.Path): Path to write the output file
        """
        # Initialize tokenizer
        enc = tiktoken.get_encoding("cl100k_base")
        
        # Read needles
        needles = []
        with open(needles_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    needle = json.loads(line)
                    needle['token_count'] = len(enc.encode(needle.get('text', '')))
                    needles.append(needle)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse needle JSON")
                    continue
        
        print(f"Found {len(needles)} needles")
        
        # Read haystack
        haystack_items = []
        with open(haystack_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    haystack_items.append(item)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse haystack JSON")
                    continue
        
        print(f"Found {len(haystack_items)} haystack items")
        
        # Get min and max token sizes
        min_tokens = min(item['token_count'] for item in haystack_items)
        max_tokens = max(item['token_count'] for item in haystack_items)
        print(f"Haystack token range: {min_tokens} to {max_tokens}")
        
        # Calculate depth ranges
        num_depths = 5  # Number of different depths to test
        depth_ranges = []
        for i in range(num_depths):
            start_idx = int((i / num_depths) * len(haystack_items))
            end_idx = int(((i + 1) / num_depths) * len(haystack_items))
            depth_ranges.append((start_idx, end_idx))
        
        # Insert needles at different depths
        output_items = []
        for needle_idx, needle in enumerate(needles):
            depth_idx = needle_idx % num_depths
            start_idx, end_idx = depth_ranges[depth_idx]
            
            # Find a suitable haystack item for this depth
            for haystack_idx in range(start_idx, end_idx):
                haystack_item = haystack_items[haystack_idx]
                text = haystack_item.get('text', '')
                
                # Calculate insertion point based on depth
                tokens = enc.encode(text)
                insert_pos = int((depth_idx + 1) / (num_depths + 1) * len(tokens))
                
                # Insert needle
                needle_text = needle.get('needle', '')
                needle_tokens = enc.encode(needle_text)
                for token in needle_tokens:
                    tokens.insert(insert_pos, token)
                    insert_pos += 1
                
                # Create new item with inserted needle
                new_item = haystack_item.copy()
                new_item['text'] = enc.decode(tokens)
                new_item['needle'] = needle
                new_item['insertion_depth'] = depth_idx
                new_item['token_count'] = len(tokens)
                
                output_items.append(new_item)
                break
        
        # Write output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in output_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Created output file with {len(output_items)} items at {output_file}")

    

    def load(self) -> BenchmarkData:
        ds_uri = 'https://github.com/open-compass/opencompass/files/14741330/needlebench.zip'
        self.download_and_unzip(ds_uri, self.dataset_path)

        haystack_files = [
            self.dataset_path / "needlebench" / "PaulGrahamEssays.jsonl",
            self.dataset_path / "needlebench" / "zh_finance.jsonl",
            self.dataset_path / "needlebench" / "zh_game.jsonl",
            self.dataset_path / "needlebench" / "zh_general.jsonl",
            self.dataset_path / "needlebench" / "zh_government.jsonl",
            self.dataset_path / "needlebench" / "zh_movie.jsonl",
            self.dataset_path / "needlebench" / "zh_tech.jsonl",
        ]

        output_file = self.dataset_path / "needlebench" / "haystack_sorted.jsonl"
        self.create_sorted_haystack(haystack_files, output_file)
        
        # Insert needles into haystack
        needles_file = self.dataset_path / "needlebench" / "needles.jsonl"
        output_with_needles = self.dataset_path / "needlebench" / "haystack_with_needles.jsonl"
        self.insert_needles_into_haystack(needles_file, output_file, output_with_needles)

        # Generate corpus, queries and relevant documents
        seen_questions = set()  # Track unique questions
        with open(output_with_needles, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    doc_id = str(len(self.corpus))
                    # Store corpus item as a dictionary with title and text
                    self.corpus[doc_id] = {
                        'title': item.get('title', ''),
                        'text': item['text']
                    }
                    
                    # Create query from needle
                    if 'needle' in item:
                        needle = item['needle']
                        question = needle.get('retrieval_question', '')
                        
                        # Skip if we've seen this question before
                        if question in seen_questions:
                            continue
                            
                        seen_questions.add(question)
                        query_id = f"q_{doc_id}"
                        self.queries[query_id] = question
                        # Store relevant docs in BEIR format: {doc_id: relevance_score}
                        self.relevant_docs[query_id] = {doc_id: 1}  # Set relevance score to 1 for relevant docs
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON from {output_with_needles}")
                    continue

        # Write corpus, queries and relevant documents to JSON files
        output_dir = self.dataset_path / "needlebench" / "retrieval_format"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write corpus
        with open(output_dir / "corpus.json", 'w', encoding='utf-8') as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2)

        # Write queries
        with open(output_dir / "queries.json", 'w', encoding='utf-8') as f:
            json.dump(self.queries, f, ensure_ascii=False, indent=2)

        # Write relevant documents (qrels)
        with open(output_dir / "qrels.json", 'w', encoding='utf-8') as f:
            json.dump(self.relevant_docs, f, ensure_ascii=False, indent=2)

        print(f"Wrote processed files to {output_dir}")

        return BenchmarkData(
            corpus=self.corpus,
            queries=self.queries,
            relevant_docs=self.relevant_docs
        )
