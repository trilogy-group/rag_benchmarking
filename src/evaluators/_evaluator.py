from typing import Dict, Any
from retrievers._retriever import Retriever

class Evaluator:
    def evaluate(self, retriever: Retriever, dataset: str) -> Dict[str, Any]:
        """
        Evaluates the retriever on the given dataset.
        Returns a dictionary of metrics: { "nDCG@10": 0.56, "MRR@10": 0.42, ... }
        """
        raise NotImplementedError

    def name(self) -> str:
        """
        Returns the evaluator name (e.g., "BEIR", "MTEB", "FRAME").
        """
        raise NotImplementedError
