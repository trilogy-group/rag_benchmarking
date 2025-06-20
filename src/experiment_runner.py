# Run experiments
from retrievers._retriever import Retriever
from evaluators._evaluator import Evaluator
from benchmark_datasets._benchmark_dataset import BenchmarkDataset
from datastores._datastore import DataStore
import logging

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, dataset: BenchmarkDataset, datastore: DataStore, retriever: Retriever, evaluator: Evaluator, max_corpus_size: int = 200000):
        self.max_corpus_size = max_corpus_size
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
        logger.info(f"Dataset loaded: {len(self.dataset.corpus)}")

        # Index the corpus
        corpus_items = list(self.dataset.corpus.items())

        # print(f"Corpus items: Item 0:{corpus_items[0]}, \n Item 1: {corpus_items[1]} \n Count: {len(corpus_items)}")

        # Prepare documents for indexing
        documents = [
            {
                "id": doc_id,
                "content": f"{doc.get('title', '')}. {doc.get('text', '')}"
            }
            for doc_id, doc in corpus_items[:self.max_corpus_size]
        ]
        logger.debug(f"Documents: {documents[0]} {len(documents)}")
        
        # print("Sample query keys:", list(self.dataset.queries.keys())[:5])
        
        first_key = next(iter(self.dataset.queries))
        logger.debug(f"Queries: {self.dataset.queries[first_key]} {len(self.dataset.queries)}")
        logger.debug(f"Relevant docs: {list(self.dataset.relevant_docs.values())[:1]}")

        # print(f"Sample query IDs: {list(self.dataset.queries.keys())[:5]}")
        # Index documents
        logger.info(f"Corpus size: {len(self.dataset.corpus)}")
        logger.info(f"Indexing documents: {self.max_corpus_size}")
        self.datastore.index_corpus(documents[:self.max_corpus_size])
        
        # Evaluate the retriever
        # self.evaluator.evaluate(self.retriever, self.dataset, self.max_corpus_size)









