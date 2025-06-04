from typing import List, Tuple

class Retriever:
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Executes a retrieval for the query. Returns list of (text, score).
        """
        raise NotImplementedError

    def index_corpus(self, documents: List[str], ids: List[str] = None) -> None:
        """
        Indexes a corpus for retrieval (optional for streaming pipelines).
        """
        raise NotImplementedError

    def name(self) -> str:
        """
        Returns the retriever name or pipeline label.
        """
        raise NotImplementedError
    
