from experiment_runner import ExperimentRunner
from benchmark_datasets.beir_dataset import BeirDataset
from benchmark_datasets.ragtruth_dataset import RagTruthDataset
from datastores.azure_ai_search_store import AzureAISearchStore
from retrievers.azure_ai_search_retriever import AzureAISearchRetriever
from evaluators.beir_evaluator import BEIREvaluator
from dotenv import load_dotenv

load_dotenv()

experiments = [
    {
        "index_name": "beir-covid-index",
        "agent_name": "beir-covid-agent",
        "dataset_name": "trec-covid",
        "k_values": [1, 3, 5, 10],
        "retriever": AzureAISearchRetriever,
        "evaluator": BEIREvaluator,
        "datastore": AzureAISearchStore,
        "dataset": BeirDataset,
        "experiment_runner": ExperimentRunner,
        "k_values": [1, 3, 5, 10],
        "retriever": AzureAISearchRetriever,
        "text_embedding_model": "text-embedding-3-large",
        "openai_model": "gpt-4o",
        "max_corpus_size": 200000,
    },
    {
        "index_name": "beir-scidocs-index",
        "agent_name": "beir-scidocs-agent",
        "dataset_name": "scidocs",
        "k_values": [1, 3, 5, 10],
        "retriever": AzureAISearchRetriever,
        "evaluator": BEIREvaluator,
        "datastore": AzureAISearchStore,
        "dataset": BeirDataset,
        "experiment_runner": ExperimentRunner,
        "k_values": [1, 3, 5, 10],
        "retriever": AzureAISearchRetriever,
        "text_embedding_model": "text-embedding-3-large",
        "openai_model": "gpt-4o",
        "max_corpus_size": 25000,
    },
    {
        "index_name": "beir-scidocs-index-ada-002-gpt4o",
        "agent_name": "beir-scidocs-agent-ada-002-gpt4o",
        "dataset_name": "scidocs",
        "k_values": [1, 3, 5, 10],
        "retriever": AzureAISearchRetriever,
        "evaluator": BEIREvaluator,
        "datastore": AzureAISearchStore,
        "dataset": BeirDataset,
        "experiment_runner": ExperimentRunner,
        "k_values": [1, 3, 5, 10],
        "retriever": AzureAISearchRetriever,
        "text_embedding_model": "text-embedding-ada-002",
        "openai_model": "gpt-4o",
        "max_corpus_size": 25000,
    },
    {
        "index_name": "ragtruth-index",
        "agent_name": "ragtruth-agent",
        "dataset_name": "ragtruth",
        "retriever": AzureAISearchRetriever,
        "evaluator": BEIREvaluator,
        "datastore": AzureAISearchStore,
        "dataset": RagTruthDataset,
        "experiment_runner": ExperimentRunner,
        "k_values": [1, 3, 5, 10],
        "text_embedding_model": "text-embedding-3-large",
        "openai_model": "gpt-4o",
        "max_corpus_size": 1000,
    }
]

def main():
    experiment = experiments[2]
    index_name = experiment["index_name"]
    agent_name = experiment["agent_name"]
    datastore = experiment["datastore"](index_name=index_name, agent_name=agent_name, text_embedding_model=experiment["text_embedding_model"], openai_model=experiment["openai_model"])
    dataset = experiment["dataset"](dataset_name=experiment["dataset_name"])
    retriever = experiment["retriever"](index_name=index_name, agent_name=agent_name)
    evaluator = experiment["evaluator"](outfile_name=f"{experiment['dataset_name']}_{experiment['text_embedding_model']}_{experiment['openai_model']}.csv")
    

    print(f"Dataset: {dataset}")
    experiment_runner = ExperimentRunner(datastore=datastore, dataset=dataset, retriever=retriever, evaluator=evaluator)
    experiment_runner.run()


if __name__ == "__main__":
    main()
