from typing import List, Tuple, Dict
from pydantic import BaseModel





class BenchmarkData(BaseModel):
    corpus: dict
    queries: dict
    relevant_docs: dict

class BenchmarkDataset:
    def get_embeddings_path(self, embedding_model: str) -> str:
        return self.base_path / f"{self.dataset_name}/embeddings/{embedding_model}.jsonl"

    def load(self) -> BenchmarkData:
        """
        Loads or prepares the dataset for use.
        """
        pass

    def get_queries(self) -> List[str]:
        """
        Returns a list of queries to test.
        """
        raise NotImplementedError

    def get_relevant_docs(self, query: str) -> List[str]:
        """
        Given a query, returns the list of relevant documents (ground truth).
        """
        raise NotImplementedError

    def get_corpus(self) -> Dict[str, Dict[str, str]]:
        return self.corpus

    def get_queries(self) -> Dict[str, str]:
        return self.queries

    def get_relevant_docs(self) -> Dict[str, list]:
        return self.relevant_docs

    def name(self) -> str:
        """
        Returns dataset name (e.g., "FiQA", "HotpotQA", "Needle").
        """
        raise NotImplementedError
