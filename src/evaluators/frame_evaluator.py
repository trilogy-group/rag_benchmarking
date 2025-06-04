from evaluators._evaluator import Evaluator
from retrievers._retriever import Retriever
from benchmark_datasets._benchmark_dataset import BenchmarkDataset
from typing import Dict, Any
import os
import csv

class FrameEvaluator(Evaluator):
    def __init__(self, outfile_name: str, k_values: list = [1, 3, 5, 10], results_base_path: str = "data/results"):
        self.outfile_name = outfile_name
        self.k_values = k_values
        self._name = "FRAME"
        self.results_base_path = results_base_path

    def name(self) -> str:
        return self._name

    def evaluate(self, retriever: Retriever, dataset: BenchmarkDataset) -> Dict[str, Any]:
        dataset.load()
        queries = dataset.get_queries()
        gold_docs = dataset.get_relevant_docs()

        hit_scores = {f"Hit@{k}": 0 for k in self.k_values}
        total = len(queries)

        for qid, query in queries.items():
            hits = retriever.retrieve(query, top_k=max(self.k_values))
            retrieved_ids = list(hits.keys())

            relevant_ids = {doc["id"] for doc in gold_docs.get(qid, [])}
            for k in self.k_values:
                top_k_hits = retrieved_ids[:k]
                if any(doc_id in relevant_ids for doc_id in top_k_hits):
                    hit_scores[f"Hit@{k}"] += 1

        # Normalize
        for k in self.k_values:
            hit_scores[f"Hit@{k}"] = round(hit_scores[f"Hit@{k}"] / total, 5)

        self.save_eval_results_to_csv(hit_scores)
        print("FRAME Evaluation Results:", hit_scores)

        return hit_scores

    def save_eval_results_to_csv(self, results: Dict[str, float]) -> None:
        os.makedirs(self.results_base_path, exist_ok=True)
        output_path = os.path.join(self.results_base_path, self.outfile_name)

        sorted_keys = sorted(results.keys())

        with open(output_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sorted_keys)
            writer.writeheader()
            writer.writerow({k: round(results[k], 5) for k in sorted_keys})

        print(f"✅ FRAME evaluation results saved to {output_path}")
