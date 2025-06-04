# Run experiments
from retrievers._retriever import Retriever
from evaluators._evaluator import Evaluator
from benchmark_datasets._benchmark_dataset import BenchmarkDataset
from datastores._datastore import DataStore

class ExperimentRunner:
    def __init__(self, dataset: BenchmarkDataset, datastore: DataStore, retriever: Retriever, evaluator: Evaluator, corpus_size: int = 200000):
        self.corpus_size = corpus_size
        self.datastore = datastore
        self.retriever = retriever
        self.evaluator = evaluator
        self.dataset = dataset

    def run(self):
        """
        Run the experiment
        """
        # Load the dataset
        self.dataset.load()
        print(f"Dataset loaded: {len(self.dataset.corpus)}")

        # Index the corpus
        corpus_items = list(self.dataset.corpus.items())

        # Prepare documents for indexing
        documents = [
            {
                "id": doc_id,
                "content": f"{doc.get('title', '')}. {doc.get('text', '')}"
            }
            for doc_id, doc in corpus_items[:self.corpus_size]
        ]
        print(f"Documents: {documents[0]} {len(documents)}")
        print(f"Queries: {self.dataset.queries['0']} {len(self.dataset.queries)}")
        print(f"Relevant docs: {self.dataset.relevant_docs['0']} {len(self.dataset.relevant_docs)}")
        # print(f"Sample query IDs: {list(self.dataset.queries.keys())[:5]}")


        # Index documents
        # self.datastore.index_corpus(documents[:self.corpus_size])
        
        # Evaluate the retriever
        # self.evaluator.evaluate(self.retriever, self.dataset)









