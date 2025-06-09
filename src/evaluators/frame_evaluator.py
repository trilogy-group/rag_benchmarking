import os
import csv
from typing import Dict, Any, Tuple
from beir.retrieval.evaluation import EvaluateRetrieval
from evaluators._evaluator import Evaluator
from retrievers._retriever import Retriever
from benchmark_datasets._benchmark_dataset import BenchmarkDataset


class FrameEvaluator(Evaluator):
    def __init__(self, outfile_name: str, k_values: list = [1, 3, 5, 10], results_base_path: str = "data/results"):
        self.outfile_name = outfile_name
        self.k_values = k_values
        self._name = "FRAME"
        self.results_base_path = results_base_path

    def name(self) -> str:
        return self._name

    def evaluate(self, retriever: Retriever, dataset: BenchmarkDataset, max_doc_id: int = None) -> Dict[str, Any]:
        # Load the benchmark dataset
        dataset.load()
        queries = dataset.get_queries()
        corpus = dataset.get_corpus()
        qrels = dataset.get_relevant_docs()

        print(f"📊 Frame Evaluating {len(queries)} queries over {len(corpus)} documents")

        # Run retrieval and collect results
        results = {}
        for qid, query in list(queries.items())[:50]:
            print(f"Query: {qid} {query}")
            hits = retriever.retrieve(query, top_k=max(self.k_values))  # {doc_id: score}
            print(f"Hits: {hits}")
            results[qid] = hits

        print(f"Results: {results} {results[list(results.keys())[0]]}")

        # Use BEIR's evaluator
        evaluator = EvaluateRetrieval()        

        metrics = evaluator.evaluate(
            qrels=qrels,
            results=results,
            k_values=self.k_values
        )

        self.save_eval_results_to_csv(metrics)
        print("✅ FRAME evaluation metrics:", metrics)
        
        return metrics

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
