from pydantic import BaseModel
from typing import Type, List

from benchmark_datasets.beir_dataset import BeirDataset

from datastores.azure_ai_search_store import AzureAISearchStore
from retrievers.azure_ai_search_retriever import AzureAISearchRetriever
from evaluators.beir_evaluator import BEIREvaluator
from evaluators.frame_evaluator import FrameEvaluator
from benchmark_datasets.frame_dataset import FrameDataset
from benchmark_datasets.ragtruth_dataset import RagTruthDataset
from experiment_runner import ExperimentRunner

class ExperimentConfig(BaseModel):
    name: str
    index_name: str
    agent_name: str
    dataset_name: str
    k_values: List[int]
    retriever: Type
    evaluator: Type
    datastore: Type
    dataset: Type
    experiment_runner: Type
    text_embedding_model: str
    openai_model: str
    max_corpus_size: int

experiments: List[ExperimentConfig] = [
    ExperimentConfig(
        name="beir-covid-3-large-gpt-4o",
        index_name="beir-covid-index",
        agent_name="beir-covid-agent",
        dataset_name="trec-covid",
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=BEIREvaluator,
        datastore=AzureAISearchStore,
        dataset=BeirDataset,
        experiment_runner=ExperimentRunner,
        text_embedding_model="text-embedding-3-large",
        openai_model="gpt-4o",
        max_corpus_size=200000
    ),
    ExperimentConfig(
        name="beir-scidocs-3-large-gpt-4o",
        index_name="beir-scidocs-index",
        agent_name="beir-scidocs-agent",
        dataset_name="scidocs",
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=BEIREvaluator,
        datastore=AzureAISearchStore,
        dataset=BeirDataset,
        experiment_runner=ExperimentRunner,
        text_embedding_model="text-embedding-3-large",
        openai_model="gpt-4o",
        max_corpus_size=25000
    ),
    ExperimentConfig(
        name="beir-scidocs-ada-002-gpt-4o",
        index_name="beir-scidocs-index-ada-002-gpt4o",
        agent_name="beir-scidocs-agent-ada-002-gpt4o",
        dataset_name="scidocs",
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=BEIREvaluator,
        datastore=AzureAISearchStore,
        dataset=BeirDataset,
        experiment_runner=ExperimentRunner,
        text_embedding_model="text-embedding-ada-002",
        openai_model="gpt-4o",
        max_corpus_size=25000
    ),
    ExperimentConfig(
        name="frame-3-large-gpt-4o",
        index_name="frame-index",
        agent_name="frame-agent",
        dataset_name="default",
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=FrameEvaluator,
        datastore=AzureAISearchStore,
        dataset=FrameDataset,
        experiment_runner=ExperimentRunner,
        text_embedding_model="text-embedding-3-large",
        openai_model="gpt-4o",
        max_corpus_size=25000
    ),
    ExperimentConfig(
        name="ragtruth-3-large-gpt-4o",
        index_name="ragtruth-index",
        agent_name="ragtruth-agent",
        dataset_name="default",  # or whichever subset you want to use
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=BEIREvaluator,
        datastore=AzureAISearchStore,
        dataset=RagTruthDataset,  # <- Use the new dataset class
        experiment_runner=ExperimentRunner,
        text_embedding_model="text-embedding-3-large",
        openai_model="gpt-4o",
        max_corpus_size=20000
    )
]

def get_experiment_config(name: str) -> ExperimentConfig:
    for exp in experiments:
        if exp.name == name:
            return exp
    raise ValueError(f"No experiment found with name: {name}")