from benchmark_datasets._benchmark_dataset import BenchmarkDataset, BenchmarkData
import pathlib
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

class BeirDataset(BenchmarkDataset):
    def __init__(self, dataset_name:str="trec-covid", base_path:str="./data/benchmark_datasets/beir"):
        self.dataset_name = dataset_name
        self.queries = []
        self.relevant_docs = []
        self.corpus = []

        self.dataset_name = dataset_name
        self.base_path = pathlib.Path(base_path)
        self.dataset_path = self.base_path / f"{self.dataset_name}"


    def load(self) -> BenchmarkData:
        """
        Loads or prepares the dataset for use.
        """
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        print(f"Downloading dataset {self.dataset_name} from {url}")
        if not os.path.exists(self.base_path / f"{self.dataset_name}.zip"):
            util.download_and_unzip(url, self.base_path)
        
        print(f"Dataset {self.dataset_name} downloaded and unzipped to {self.dataset_path} ")

        # GenericDataLoader.load() returns a tuple of (corpus, queries, qrels)
        self.corpus, self.queries, self.relevant_docs = GenericDataLoader(data_folder=str(self.dataset_path)).load(split="test")
        print(f"Loaded corpus with {len(self.corpus)} documents")
        print(f"Loaded {len(self.queries)} queries")
        print(f"Loaded {len(self.relevant_docs)} relevant document mappings")

        # Print the first document
        if self.corpus:
            first_doc_id = next(iter(self.corpus))
            first_doc = self.corpus[first_doc_id]
            print(f"First document ID: {first_doc_id}")
            print(f"First document content: {first_doc}")
            print(f"First document keys: {list(first_doc.keys())}")

            first_doc_relevant_queries = [query for query, doc_ids in self.relevant_docs.items() if first_doc_id in doc_ids]
            print(f"Queries related to the first document: {first_doc_relevant_queries} {len(first_doc_relevant_queries)}")

            first_doc_related_docs = [self.corpus[doc_id] for doc_id in self.relevant_docs.get(first_doc_id, [])]
            print(f"Documents related to the first document: {first_doc_related_docs} {len(first_doc_related_docs)}")

        # print the first query
        if self.queries:
            first_query_id = next(iter(self.queries))
            first_query = self.queries[first_query_id]
            print(f"First query ID: {first_query_id}")
            print(f"First query content: {first_query}")
            print(f"First query keys: {list(first_query)}")

            first_query_relevant_docs = [doc_id for doc_id in self.relevant_docs.get(first_query_id, [])]
            print(f"Documents related to the first query: {first_query_relevant_docs[0]} {len(first_query_relevant_docs)}")

        return BenchmarkData(
            corpus=self.corpus,
            queries=self.queries,
            relevant_docs=self.relevant_docs
        )

    def get_corpus(self) -> dict:
        return self.corpus
    
    def get_queries(self) -> dict:
        return self.queries
    
    def get_relevant_docs(self) -> dict:
        return self.relevant_docs




        
        
        
    

        
