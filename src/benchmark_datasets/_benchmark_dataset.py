from typing import List, Tuple
from pydantic import BaseModel




class BenchmarkData(BaseModel):
    corpus: dict
    queries: dict
    relevant_docs: dict

class BenchmarkDataset:
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

    def get_corpus(self) -> List[str]:
        """
        Returns the entire document corpus.
        """
        raise NotImplementedError

    def name(self) -> str:
        """
        Returns dataset name (e.g., "FiQA", "HotpotQA", "Needle").
        """
        raise NotImplementedError
