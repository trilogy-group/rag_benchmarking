from evaluators._evaluator import Evaluator
from retrievers._retriever import Retriever
from benchmark_datasets._benchmark_dataset import BenchmarkDataset
from beir.retrieval.evaluation import EvaluateRetrieval
from typing import Dict, Any, Tuple
import csv
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class BEIREvaluator(Evaluator):
    def __init__(self, outfile_name: str, k_values: list = [1, 3, 5, 10], results_base_path: str = "data/results"):
        self.outfile_name = outfile_name
        self.k_values = k_values
        self._name = "BEIR"
        self.results_base_path = results_base_path

    # def filter_qrels_by_max_corpus_size(self, qrels: Dict[str, Dict[str, int]], max_corpus_size: int = None) -> Dict[str, Dict[str, int]]:
    #     print(f"Filtering qrels by docid: {max_corpus_size}")

    #     filtered_qrels = {
    #         qid: {
    #             docid: rel for docid, rel in doc_rels.items()
    #             if max_corpus_size is None or int(docid.replace("doc", "")) < max_corpus_size
    #         }
    #         for qid, doc_rels in qrels.items()
    #     }

    #     filtered_qrels = {qid: doc_rels for qid, doc_rels in filtered_qrels.items() if doc_rels}
    #     return filtered_qrels

    def filter_qrels_by_max_corpus_size(
        self, 
        qrels: Dict[str, Dict[str, int]], 
        max_corpus_size: int, 
        corpus: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, int]]:
        print(f"Filtering qrels using top {max_corpus_size} corpus documents")

        # Create the reduced corpus and set of allowed doc IDs
        reduced_corpus = dict(list(corpus.items())[:max_corpus_size])
        allowed_docids = set(reduced_corpus.keys())

        # Keep only queries where all qrel docids exist in the allowed set
        filtered_qrels = {
            qid: doc_rels
            for qid, doc_rels in qrels.items()
            if all(docid in allowed_docids for docid in doc_rels)
        }

        return filtered_qrels
    
    def filter_queries_by_qrels(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> Dict[str, str]:
        return {qid: queries[qid] for qid in qrels if qid in queries}


    def name(self) -> str:
        return self._name

    # def evaluate(self, retriever: Retriever, dataset: BenchmarkDataset, max_query_count: int = None, max_corpus_size: int = None) -> Dict[str, Any]:

    #     # Load the dataset if it hasn't been already
    #     if not dataset.get_corpus():
    #         dataset.load()

    #     corpus = dataset.get_corpus()
    #     queries = dataset.get_queries()
    #     qrels = dataset.get_relevant_docs()

    #     filtered_qrels = self.filter_qrels_by_max_corpus_size(qrels, corpus=corpus, max_corpus_size=max_corpus_size)
    #     filtered_queries = self.filter_queries_by_qrels(queries, filtered_qrels)

    #     print(f"Original query count: {len(queries)} queries")
    #     print(f"Corpus size: {max_corpus_size} docs, {len(filtered_queries)} matching queries")
    #     print(f"Query count for evaluation: {max_query_count} queries")

    #     # Retrieve results using the provided retriever
    #     results = {}
    #     for query_id, query in list(filtered_queries.items())[:max_query_count]:
    #         hits = retriever.retrieve(query, top_k=max(self.k_values))
    #         results[query_id] = hits
    #         print(f"Query {query_id}: {query}")
    #         print(f"Hits: {hits}")

    #     # Use BEIR's evaluation utility
    #     evaluator = EvaluateRetrieval()
    #     scores = evaluator.evaluate(qrels, results, k_values=self.k_values)

    #     self.save_eval_results_to_csv(scores)

    #     print("Evaluation Results:", scores)
    #     # for metric, value in scores.items():
    #     #     print(f"{metric}: {value:.4f}")
    #     return scores

    def evaluate(
        self,
        retriever: Retriever,
        dataset: BenchmarkDataset,
        max_query_count: int = None,
        max_corpus_size: int = None,
        batch_size: int = 20
    ) -> Dict[str, Any]:

        if not dataset.get_corpus():
            dataset.load()

        corpus = dataset.get_corpus()
        queries = dataset.get_queries()
        qrels = dataset.get_relevant_docs()

        filtered_qrels = self.filter_qrels_by_max_corpus_size(qrels, corpus=corpus, max_corpus_size=max_corpus_size)
        filtered_queries = self.filter_queries_by_qrels(queries, filtered_qrels)

        print(f"Original query count: {len(queries)}")
        print(f"Corpus size: {max_corpus_size}")
        print(f"Filtered queries: {len(filtered_queries)}")
        print(f"Evaluating with batch_size={batch_size}, max_query_count={max_query_count}")

        query_items = list(filtered_queries.items())[:max_query_count]
        results = {}

        total_batches = (len(query_items) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(query_items), batch_size), desc="Evaluating", unit="batch", total=total_batches):
            batch = query_items[i:i+batch_size]
            query_ids = [qid for qid, _ in batch]
            texts = [query for _, query in batch]

            if hasattr(retriever, "batch_retrieve"):
                # Native batch retrieve
                batch_results = retriever.batch_retrieve(texts, top_k=max(self.k_values))
            else:
                # Parallel single-query retrieval using ThreadPool
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
                            print(f"❌ Error in retrieval for query {query_ids[idx]}: {e}")
                            batch_results[idx] = []

            for qid, hits in zip(query_ids, batch_results):
                results[qid] = hits

        evaluator = EvaluateRetrieval()
        scores = evaluator.evaluate(filtered_qrels, results, k_values=self.k_values)
        print(f"Scores: {scores}")

        self.save_eval_results_to_csv(scores)
        print("✅ Evaluation complete.")
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
