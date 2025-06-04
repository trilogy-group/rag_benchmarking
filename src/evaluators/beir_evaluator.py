from evaluators._evaluator import Evaluator
from retrievers._retriever import Retriever
from benchmark_datasets._benchmark_dataset import BenchmarkDataset
from beir.retrieval.evaluation import EvaluateRetrieval
from typing import Dict, Any, Tuple
import csv
import os


class BEIREvaluator(Evaluator):
    def __init__(self, outfile_name: str, k_values: list = [1, 3, 5, 10], results_base_path: str = "data/results"):
        self.outfile_name = outfile_name
        self.k_values = k_values
        self._name = "BEIR"
        self.results_base_path = results_base_path

    def name(self) -> str:
        return self._name

    def evaluate(self, retriever: Retriever, dataset: BenchmarkDataset ) -> Dict[str, Any]:

        # Load the dataset if it hasn't been already
        if not dataset.get_corpus():
            dataset.load()

        corpus = dataset.get_corpus()
        queries = dataset.get_queries()
        qrels = dataset.get_relevant_docs()

        print(f"Evaluating with corpus: {len(corpus)} docs, {len(queries)} queries")

        # Retrieve results using the provided retriever
        results = {}
        for query_id, query in list(queries.items())[:100]:
            hits = retriever.retrieve(query, top_k=max(self.k_values))
            results[query_id] = hits
            print(f"Query {query_id}: {query}")
            print(f"Hits: {hits}")

        # Use BEIR's evaluation utility
        evaluator = EvaluateRetrieval()
        scores = evaluator.evaluate(qrels, results, k_values=self.k_values)

        self.save_eval_results_to_csv(scores)

        print("Evaluation Results:", scores)
        # for metric, value in scores.items():
        #     print(f"{metric}: {value:.4f}")

        return scores
    
    def save_eval_results_to_csv(self,
        results: Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]],
    ) -> None:
        
        os.makedirs(self.results_base_path, exist_ok=True)

        output_path = os.path.join(self.results_base_path, self.outfile_name)
        ndcg, map_, recall, precision = results

        # Merge all metrics into a single flat dictionary
        all_metrics = {}
        all_metrics.update(ndcg)
        all_metrics.update(map_)
        all_metrics.update(recall)
        all_metrics.update(precision)

        # Optional: sort keys for consistent column order
        sorted_keys = sorted(all_metrics.keys())

        # Write to CSV
        with open(output_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sorted_keys)
            writer.writeheader()
            writer.writerow({k: round(all_metrics[k], 5) for k in sorted_keys})

        print(f"✅ Evaluation results saved to {output_path}")
