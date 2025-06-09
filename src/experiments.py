from pydantic import BaseModel
from typing import Type, List, Optional
from enum import Enum
from benchmark_datasets.beir_dataset import BeirDataset
from benchmark_datasets._benchmark_dataset import BenchmarkDataset
from datastores.azure_ai_search_store import AzureAISearchStore
from retrievers.azure_ai_search_retriever import AzureAISearchRetriever
from benchmark_datasets.frame_dataset import FrameDataset
from benchmark_datasets.ragtruth_dataset import RagTruthDataset
from experiment_runner import ExperimentRunner
from datastores.pinecone_datastore import PineconeDatastore
from retrievers.pinecone_retriever import PineconeRetriever
from datastores._datastore import DataStore
from evaluators._evaluator import Evaluator
from evaluators.beir_evaluator import BEIREvaluator
from evaluators.frame_evaluator import FrameEvaluator
from retrievers._retriever import Retriever
from llm_models import OpenAIModel, AnthropicModel
from datastores.qdrant_datastore import QdrantDatastore
from retrievers.qdrant_retriever import QdrantRetriever
from embeddings.embedding_models import CohereEmbeddingModel, GeminiEmbeddingModel, PineconeNativeEmbeddingModel, OpenAIEmbeddingModel, StellaEmbeddingModel, ModernBERTEmbeddingModel, IntfloatEmbeddingModel
from benchmark_datasets.hotpotqa_dataset import HotpotQADataset
from datastores.vertex_ai_datastore import VertexAiDatastore
from retrievers.vertex_ai_retriever import VertexAIRetriever

class ExperimentName(Enum):
    AZURE_BEIR_COVID_3_LARGE_GPT_4O = "azure-beir-covid-3-large-gpt-4o"
    AZURE_BEIR_SCIDOCS_3_LARGE_GPT_4O = "azure-beir-scidocs-3-large-gpt-4o"
    AZURE_BEIR_SCIDOCS_ADA_002_GPT_4O = "azure-beir-scidocs-ada-002-gpt-4o"
    AZURE_FRAME_3_LARGE_GPT_4O = "azure-frame-3-large-gpt-4o"
    AZURE_RAGTRUTH_3_LARGE_GPT_4O = "azure-ragtruth-3-large-gpt-4o"
    AZURE_HOTPOTQA_3_LARGE_GPT_4O = "azure-hotpotqa-3-large-gpt-4o"
    PINECONE_FRAME_LLAMAV2_GPT_4O = "pinecone-frame-llamav2-gpt-4o"
    PINECONE_FRAME_E5LARGE_GPT_4O = "pinecone-frame-e5large-gpt-4o"
    PINECONE_BEIR_COVID_LLAMAV2_GPT_4O = "pinecone-beir-covid-llamav2-gpt-4o"
    PINECONE_BEIR_SCIDOCS_LLAMAV2_GPT_4O = "pinecone-beir-scidocs-llamav2-gpt-4o"
    PINECONE_BEIR_SCIDOCS_E5LARGE_GPT_4O = "pinecone-beir-scidocs-e5large-gpt-4o"    
    PINECONE_BEIR_NQ_3_LARGE_GPT_4O = "pinecone-beir-nq-3-large-gpt-4o"
    PINECONE_BEIR_NQ_GEMINI_001_GPT_4O = "pinecone-beir-nq-gemini-001-gpt-4o"
    PINECONE_FRAME_3_LARGE_GPT_4O = "pinecone-frame-3-large-gpt-4o"
    PINECONE_FRAME_COHERE_V4_GPT_4O = "pinecone-frame-cohere-v4-gpt-4o"
    PINECONE_FRAME_GEMINI_001_GPT_4O = "pinecone-frame-gemini-001-gpt-4o"
    PINECONE_FRAME_GEMINI_EXP_03_07_GPT_4O = "pinecone-frame-gemini-exp-03-07-gpt-4o"
    PINECONE_FRAME_STELLA_1_5B_V5_GPT_4O = "pinecone-frame-stella-1-5b-v5-gpt-4o"
    PINECONE_FRAME_MODERN_BERT_LARGE_GPT_4O = "pinecone-frame-modern-bert-large-gpt-4o"
    QDRANT_BEIR_COVID_3_LARGE_GPT_4O = "qdrant-beir-covid-3-large-gpt-4o"
    QDRANT_BEIR_SCIDOCS_3_LARGE_GPT_4O = "qdrant-beir-scidocs-3-large-gpt-4o"
    QDRANT_BEIR_SCIDOCS_COHERE_V4_GPT_4O = "qdrant-beir-scidocs-cohere-v4-gpt-4o"
    QDRANT_FRAME_3_LARGE_GPT_4O = "qdrant-frame-3-large-gpt-4o"
    QDRANT_FRAME_COHERE_V4_GPT_4O = "qdrant-frame-cohere-v4-gpt-4o"
    QDRANT_FRAME_GEMINI_001_GPT_4O = "qdrant-frame-gemini-001-gpt-4o"
    QDRANT_FRAME_GEMINI_EXP_03_07_GPT_4O = "qdrant-frame-gemini-exp-03-07-gpt-4o"
    QDRANT_FRAME_STELLA_1_5B_V5_GPT_4O = "qdrant-frame-stella-1-5b-v5-gpt-4o"
    QDRANT_FRAME_MODERN_BERT_LARGE_GPT_4O = "qdrant-frame-modern-bert-large-gpt-4o"
    QDRANT_HOTPOTQA_3_LARGE_GPT_4O = "qdrant-hotpotqa-3-large-gpt-4o"
    QDRANT_HOTPOTQA_GEMINI_001_GPT_4O = "qdrant-hotpotqa-gemini-001-gpt-4o"
    VERTEX_AI_FRAME_TEXT_EMBEDDING_005_GPT_4O = "vertex-ai-frame-text-embedding-005-gpt-4o"
    VERTEX_AI_FRAME_ML_E5_LARGE_GPT_4O = "vertex-ai-frame-ml-e5-large-gpt-4o"
    

class ExperimentConfig(BaseModel):
    name: ExperimentName
    dataset_name: str
    k_values: List[int]
    retriever: Type[Retriever]
    evaluator: Type[Evaluator]
    datastore: Type[DataStore]
    dataset: Type[BenchmarkDataset]
    text_embedding_model: str
    openai_model: str
    max_corpus_size: int

    class Config:
        arbitrary_types_allowed = True



experiments: List[ExperimentConfig] = [
    ExperimentConfig(
        name=ExperimentName.AZURE_BEIR_COVID_3_LARGE_GPT_4O,
        dataset_name="trec-covid",
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=BEIREvaluator,
        datastore=AzureAISearchStore,
        dataset=BeirDataset,
        experiment_runner=ExperimentRunner,
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=200000
    ),
    ExperimentConfig(
        name=ExperimentName.AZURE_BEIR_SCIDOCS_3_LARGE_GPT_4O,
        dataset_name="scidocs",
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=BEIREvaluator,
        datastore=AzureAISearchStore,
        dataset=BeirDataset,
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000
    ),
    ExperimentConfig(
        name=ExperimentName.AZURE_BEIR_SCIDOCS_ADA_002_GPT_4O,
        dataset_name="scidocs",
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=BEIREvaluator,
        datastore=AzureAISearchStore,
        dataset=BeirDataset,
        text_embedding_model="text-embedding-ada-002",
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000
    ),
    ExperimentConfig(
        name=ExperimentName.AZURE_FRAME_3_LARGE_GPT_4O,
        dataset_name="default",
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=FrameEvaluator,
        datastore=AzureAISearchStore,
        dataset=FrameDataset,
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000
    ),
    ExperimentConfig(
        name=ExperimentName.AZURE_RAGTRUTH_3_LARGE_GPT_4O,
        dataset_name="default",  # or whichever subset you want to use
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=BEIREvaluator,
        datastore=AzureAISearchStore,
        dataset=RagTruthDataset,  # <- Use the new dataset class
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=20000
    ),
    ExperimentConfig(
        name=ExperimentName.AZURE_HOTPOTQA_3_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=AzureAISearchRetriever,
        evaluator=FrameEvaluator,
        datastore=AzureAISearchStore,
        dataset=HotpotQADataset,
        dataset_name="test",
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=100,
        corpus_size=100,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_FRAME_LLAMAV2_GPT_4O.value,
        dataset_name="default",
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=FrameEvaluator,
        datastore=PineconeDatastore,
        dataset=FrameDataset,
        text_embedding_model=PineconeNativeEmbeddingModel.LLAMA_V2.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_BEIR_COVID_LLAMAV2_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=BEIREvaluator,
        datastore=PineconeDatastore,
        dataset=BeirDataset,
        dataset_name="trec-covid",
        text_embedding_model=PineconeNativeEmbeddingModel.LLAMA_V2.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=200000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_BEIR_NQ_3_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=BEIREvaluator,
        datastore=PineconeDatastore,
        dataset=BeirDataset,
        dataset_name="nq",
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
        corpus_size=200000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_BEIR_NQ_GEMINI_001_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=BEIREvaluator,
        datastore=PineconeDatastore,
        dataset=BeirDataset,
        dataset_name="nq",
        text_embedding_model=GeminiEmbeddingModel.GEMINI_001.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=200000,
        corpus_size=200000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_BEIR_SCIDOCS_LLAMAV2_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=BEIREvaluator,
        datastore=PineconeDatastore,
        dataset=BeirDataset,
        dataset_name="scidocs",
        text_embedding_model=PineconeNativeEmbeddingModel.LLAMA_V2.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_FRAME_E5LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=FrameEvaluator,
        datastore=PineconeDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=PineconeNativeEmbeddingModel.MULTILINGUAL_E5_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_BEIR_SCIDOCS_E5LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=BEIREvaluator,
        datastore=PineconeDatastore,
        dataset=BeirDataset,
        dataset_name="scidocs",
        text_embedding_model=PineconeNativeEmbeddingModel.MULTILINGUAL_E5_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_FRAME_3_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=FrameEvaluator,
        datastore=PineconeDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_FRAME_COHERE_V4_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=FrameEvaluator,
        datastore=PineconeDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=CohereEmbeddingModel.COHERE_EMBEDDING_V4.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_FRAME_GEMINI_001_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=FrameEvaluator,
        datastore=PineconeDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=GeminiEmbeddingModel.GEMINI_001.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_FRAME_GEMINI_EXP_03_07_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=FrameEvaluator,
        datastore=PineconeDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=GeminiEmbeddingModel.GEMINI_EXP_03_07.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.PINECONE_FRAME_STELLA_1_5B_V5_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=PineconeRetriever,
        evaluator=FrameEvaluator,
        datastore=PineconeDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=StellaEmbeddingModel.STELLA_1_5B_V5.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),

    ExperimentConfig(
        name=ExperimentName.QDRANT_BEIR_COVID_3_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=BEIREvaluator,
        datastore=QdrantDatastore,
        dataset=BeirDataset,
        dataset_name="trec-covid",
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=200000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_BEIR_SCIDOCS_3_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=BEIREvaluator,
        datastore=QdrantDatastore,
        dataset=BeirDataset,
        dataset_name="scidocs",
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_BEIR_SCIDOCS_COHERE_V4_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=BEIREvaluator,
        datastore=QdrantDatastore,
        dataset=BeirDataset,
        dataset_name="scidocs",
        text_embedding_model=CohereEmbeddingModel.COHERE_EMBEDDING_V4.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_FRAME_3_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=FrameEvaluator,
        datastore=QdrantDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_FRAME_COHERE_V4_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=FrameEvaluator,
        datastore=QdrantDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=CohereEmbeddingModel.COHERE_EMBEDDING_V4.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_FRAME_GEMINI_001_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=FrameEvaluator,
        datastore=QdrantDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=GeminiEmbeddingModel.GEMINI_001.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_FRAME_GEMINI_EXP_03_07_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=FrameEvaluator,
        datastore=QdrantDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=GeminiEmbeddingModel.GEMINI_EXP_03_07.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_FRAME_STELLA_1_5B_V5_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=FrameEvaluator,
        datastore=QdrantDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=StellaEmbeddingModel.STELLA_1_5B_V5.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_FRAME_MODERN_BERT_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=FrameEvaluator,
        datastore=QdrantDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=ModernBERTEmbeddingModel.MODERN_BERT_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_HOTPOTQA_3_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=FrameEvaluator,
        datastore=QdrantDatastore,
        dataset=HotpotQADataset,
        dataset_name="test",
        text_embedding_model=OpenAIEmbeddingModel.TEXT_EMBEDDING_3_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        corpus_size=100000,
        max_corpus_size=100000,
    ),
    ExperimentConfig(
        name=ExperimentName.QDRANT_HOTPOTQA_GEMINI_001_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=QdrantRetriever,
        evaluator=FrameEvaluator,
        datastore=QdrantDatastore,
        dataset=HotpotQADataset,
        dataset_name="test",
        text_embedding_model=GeminiEmbeddingModel.GEMINI_001.value,
        openai_model=OpenAIModel.GPT_4O.value,
        corpus_size=100000,
        max_corpus_size=100000,
    ),
    ExperimentConfig(
        name=ExperimentName.VERTEX_AI_FRAME_TEXT_EMBEDDING_005_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=VertexAIRetriever,
        evaluator=FrameEvaluator,
        datastore=VertexAiDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=GeminiEmbeddingModel.TEXT_EMBEDDING_005.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
    ExperimentConfig(
        name=ExperimentName.VERTEX_AI_FRAME_ML_E5_LARGE_GPT_4O.value,
        k_values=[1, 3, 5, 10],
        retriever=VertexAIRetriever,
        evaluator=FrameEvaluator,
        datastore=VertexAiDatastore,
        dataset=FrameDataset,
        dataset_name="default",
        text_embedding_model=IntfloatEmbeddingModel.ML_E5_LARGE.value,
        openai_model=OpenAIModel.GPT_4O.value,
        max_corpus_size=25000,
    ),
]

def get_experiment_config(name: ExperimentName) -> ExperimentConfig:
    for exp in experiments:
        if exp.name == name:
            return exp
    raise ValueError(f"No experiment found with name: {name}")