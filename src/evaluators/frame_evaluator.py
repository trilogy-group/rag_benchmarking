import os
import csv
from typing import Dict, Any, Tuple
from beir.retrieval.evaluation import EvaluateRetrieval
from evaluators._evaluator import Evaluator
from retrievers._retriever import Retriever
from benchmark_datasets._benchmark_dataset import BenchmarkDataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class FrameEvaluator(Evaluator):
    def __init__(self, outfile_name: str, k_values: list = [1, 3, 5, 10], results_base_path: str = "data/results", max_query_count: int = 1000):
        self.outfile_name = outfile_name
        self.k_values = k_values
        self._name = "FRAME"
        self.results_base_path = results_base_path
        self.max_query_count = max_query_count

    def name(self) -> str:
        return self._name

    def filter_qrels_by_max_corpus_size(
        self, 
        qrels: Dict[str, Dict[str, int]], 
        max_corpus_size: int, 
        corpus: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        print(f"Filtering qrels using top {max_corpus_size} corpus documents")
        reduced_corpus = dict(list(corpus.items())[:max_corpus_size])
        allowed_docids = set(reduced_corpus.keys())
        filtered_qrels = {
            qid: doc_rels
            for qid, doc_rels in qrels.items()
            if all(docid in allowed_docids for docid in doc_rels)
        }
        return filtered_qrels

    def filter_queries_by_qrels(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> Dict[str, str]:
        return {qid: queries[qid] for qid in qrels if qid in queries}

    def evaluate(self, retriever: Retriever, dataset: BenchmarkDataset, max_query_count: int = None, max_corpus_size: int = None, batch_size: int = 20) -> Dict[str, Any]:
        dataset.load()
        corpus = dataset.get_corpus()
        queries = dataset.get_queries()
        qrels = dataset.get_relevant_docs()

        if max_corpus_size:
            qrels = self.filter_qrels_by_max_corpus_size(qrels, max_corpus_size, corpus)
            queries = self.filter_queries_by_qrels(queries, qrels)
            corpus = dict(list(corpus.items())[:max_corpus_size])

        if max_query_count is None:
            max_query_count = self.max_query_count

        query_items = list(queries.items())[:max_query_count]
        results = {}

        total_batches = (len(query_items) + batch_size - 1) // batch_size
        print(f"📊 Frame Evaluating {len(query_items)} queries over {len(corpus)} documents in {total_batches} batches")

        for i in tqdm(range(0, len(query_items), batch_size), desc="Evaluating", unit="batch"):
            batch = query_items[i:i + batch_size]
            query_ids = [qid for qid, _ in batch]
            texts = [query for _, query in batch]

            if hasattr(retriever, "batch_retrieve"):
                batch_results = retriever.batch_retrieve(texts, top_k=max(self.k_values))
            else:
                batch_results = [None] * len(texts)
                with ThreadPoolExecutor(max_workers=min(batch_size, 16)) as executor:
                    future_to_idx = {
                        executor.submit(retriever.retrieve, text, top_k=max(self.k_values)): idx
                        for idx, text in enumerate(texts)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            batch_results[idx] = future.result()
                        except Exception as e:
                            print(f"❌ Error for query {query_ids[idx]}: {e}")
                            batch_results[idx] = {}

            for qid, hits in zip(query_ids, batch_results):
                results[qid] = hits

        beir_qrels = {qid: {docid: 1 for docid in docids} for qid, docids in qrels.items()}
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(qrels=beir_qrels, results=results, k_values=self.k_values)

        self.save_eval_results_to_csv(metrics)
        print("✅ FRAME evaluation metrics:", metrics)
        return metrics

    def save_eval_results_to_csv(self, results: Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]) -> None:
        os.makedirs(self.results_base_path, exist_ok=True)
        output_path = os.path.join(self.results_base_path, self.outfile_name)

        ndcg, map_, recall, precision = results
        all_metrics = {**ndcg, **map_, **recall, **precision}
        sorted_keys = sorted(all_metrics.keys())

        with open(output_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sorted_keys)
            writer.writeheader()
            writer.writerow({k: round(all_metrics[k], 5) for k in sorted_keys})

        print(f"✅ Evaluation results saved to {output_path}")
