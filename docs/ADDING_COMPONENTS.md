# Adding New Components

This project is designed to be extensible. The sections below outline the basic steps for introducing new datasources, retrievers, benchmark datasets and embedding models.

## Adding a Datastore
1. Create a new file in `src/datastores/` and subclass `DataStore` from `_datastore.py`.
2. Implement `index_corpus()` to store documents in your backend and add any setup code such as index creation.
3. Optionally implement other helpers like `load_precomputed_embeddings()` if your datastore can read existing vectors.
4. Reference your class inside `experiments.py` when defining an `ExperimentConfig` to make it usable in runs.

## Adding a Retriever
1. Add a module under `src/retrievers/` inheriting from `Retriever` in `_retriever.py`.
2. Provide implementations of `retrieve()`, `index_corpus()` (if needed) and `name()`.
3. Import the new class in `experiments.py` and use it when configuring experiments.

## Adding a Dataset
1. Create a file in `src/benchmark_datasets/` extending `BenchmarkDataset` from `_benchmark_dataset.py`.
2. Implement a `load()` method that prepares a `BenchmarkData` object containing the corpus, queries and relevant documents.
3. Expose your dataset in `src/benchmark_datasets/__init__.py` and reference it in `experiments.py`.

## Adding an Embedding Model
1. Implement a class in `src/embeddings/` that subclasses `Embedding` from `_embedding.py`.
2. Add the model name to the relevant Enum in `embedding_models.py`.
3. Update `embedding_helper.py` so the helper can instantiate your new class when the Enum value is provided.
4. After these changes you can use the new embedding by setting `text_embedding_model` in an experiment configuration.

