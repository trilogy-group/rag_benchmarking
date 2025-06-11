from typing import Dict, Any, Tuple
from retrievers._retriever import Retriever

class Evaluator:
    def evaluate(self, retriever: Retriever, dataset: str, max_query_count:int=None, max_corpus_size:int=None) -> Dict[str, Any]:
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
    
    def save_eval_results_to_csv(self,
        results: Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]],
    ) -> None:
        """
        Saves the evaluation results to a CSV file.
        """
        raise NotImplementedError
