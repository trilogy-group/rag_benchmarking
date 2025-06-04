from typing import List, Tuple

class DataStore:
    def index(self, documents: List[str], ids: List[str] = None) -> None:
        """
        Indexes a list of documents. Optionally takes document IDs.
        """
        raise NotImplementedError

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Searches for the top_k most similar documents to the query.
        Returns a list of (document_text, score).
        """
        raise NotImplementedError

    def clear(self) -> None:
        """
        Clears all indexed documents.
        """
        raise NotImplementedError

    def name(self) -> str:
        """
        Returns the name of the vector store (for logging and metrics).
        """
        raise NotImplementedError
