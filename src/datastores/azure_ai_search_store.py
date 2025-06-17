from datastores._datastore import DataStore
from typing import List, Tuple
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, VectorSearch, VectorSearchProfile,
    HnswAlgorithmConfiguration, AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters,
    KnowledgeAgent, KnowledgeAgentTargetIndex,
    KnowledgeAgentAzureOpenAIModel, SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField
)
from azure.search.documents import SearchIndexingBufferedSender,SearchClient
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class AzureAISearchStore(DataStore):
    def __init__(self, namespace: str, index_name: str, agent_name: str, openai_model: str, text_embedding_model: str):
        self.namespace = namespace
        self.index_name = index_name
        self.agent_name = agent_name
        self.openai_model = openai_model

        logger.info(f"Index Name: {self.index_name}")
        logger.info(f"Agent Name: {self.agent_name}")
        

        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.embedding_model = text_embedding_model
        self.embedding_deployment = text_embedding_model
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.agent_client = KnowledgeAgentRetrievalClient(endpoint=self.endpoint, agent_name=self.agent_name, credential=self.credential)
        logger.info(f"Endpoint: {self.endpoint}")
        # print(f"Credential: {self.credential}")
        # print(f"OpenAI API Key: {self.azure_openai_api_key}")
        logger.info(f"OpenAI Model: {self.openai_model}")
        logger.info(f"Embedding Model: {self.embedding_model}")
        logger.debug(f"Embedding Deployment: {self.embedding_deployment}")
        logger.debug(f"OpenAI Endpoint: {self.azure_openai_endpoint}")
        self.setup_index()
        # self.setup_agent()

    def get_embedding_dimensions(self, model_name: str) -> int:
        return {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }.get(model_name, 1536)  # default to 1536 if unknown

    def setup_index(self):
        logger.info(f"Setting up index {self.index_name}")

        index = SearchIndex(
            name=self.index_name,
            fields=[
                SearchField(name="id", type="Edm.String", key=True, filterable=True),
                SearchField(name="content", type="Edm.String"),
                SearchField(name="embedding", type="Collection(Edm.Single)", stored=False,            
                vector_search_dimensions=self.get_embedding_dimensions(self.embedding_model), 
                vector_search_profile_name="hnsw_text_3_large"),
            ],
            vector_search=VectorSearch(
                # profiles=[VectorSearchProfile(name="profile", algorithm_configuration_name="alg", vectorizer_name="vec")],
                # algorithms=[HnswAlgorithmConfiguration(name="alg")],
                # vectorizers=[
                #     AzureOpenAIVectorizer(
                #         vectorizer_name="vec",
                #         parameters=AzureOpenAIVectorizerParameters(
                #             resource_url=self.azure_openai_endpoint,
                #             api_key=self.azure_openai_api_key,
                #             model_name=self.embedding_model,
                #             deployment_name=self.embedding_deployment
                #         )
                #     )
                # ]
                profiles=[VectorSearchProfile(
                        name="hnsw_text_3_large", 
                        algorithm_configuration_name="alg",
                        vectorizer_name="azure_openai_text_3_large"
                    )],
                # profiles=[VectorSearchProfile(name="profile", algorithm_configuration_name="alg", vectorizer_name="vec")],
                algorithms=[HnswAlgorithmConfiguration(name="alg")],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        vectorizer_name="azure_openai_text_3_large",
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=self.azure_openai_endpoint,
                            deployment_name=self.embedding_deployment,
                            model_name=self.embedding_model,
                            api_key=self.azure_openai_api_key
                        )
                    )
                ]
            ),
            semantic_search=SemanticSearch(
                default_configuration_name="default-semantic-config",
                configurations=[
                    SemanticConfiguration(
                        name="default-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            content_fields=[SemanticField(field_name="content")]
                        )
                    )
                ]
            )
        )
        self.index_client.create_or_update_index(index)
        logger.info(f"Index {index} created")
    
    def setup_agent(self):
        logger.info(f"Setting up agent {self.agent_name}")

        agent = KnowledgeAgent(
            name=self.agent_name,
            models=[
                KnowledgeAgentAzureOpenAIModel(
                    azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                        resource_url=self.azure_openai_endpoint,
                        api_key=self.azure_openai_api_key,
                        model_name=self.openai_model,
                        deployment_name=self.openai_model
                    )
                )
            ],
            target_indexes=[KnowledgeAgentTargetIndex(index_name=self.index_name, default_reranker_threshold=2.5)]
        )
       
        self.index_client.create_or_update_agent(agent)
        logger.info(f"Agent {self.agent_name} created")

    def index_corpus_simple(self, documents: List[str], ids: List[str] = None) -> None:
        logger.info(f"Indexing {len(documents)} documents to index {self.index_name}")
        with SearchIndexingBufferedSender(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential) as sender:
            logger.debug(f"Uploading {len(documents)} documents to index {self.index_name}")
            sender.upload_documents(documents=documents)
        logger.info(f"Documents uploaded to index {self.index_name}")
    
    def count_documents(self) -> int:
        search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)
        results = search_client.search(search_text="*")
        return len(list(results))

    def index_corpus(self, corpus: List[dict], ids: List[str] = None, batch_size: int = 100, embeddings_file_path: str = None) -> None:
        total = len(corpus)
        if total == 0:
            logger.warning("No documents to index.")
            return

        logger.info(f"Indexing {total} documents to index '{self.index_name}' in batches of {batch_size}.")

        logger.debug(f"Documents: {corpus[:3]}")

        try:
            with SearchIndexingBufferedSender(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential
            ) as sender:

                for i in tqdm(range(0, total, batch_size), desc="Uploading batches"):
                    batch = corpus[i:i + batch_size]
                    logger.debug(f"Uploading batch {i // batch_size + 1}: type={type(batch)}")
                    sender.upload_documents(batch)

            logger.info(f"All {total} documents uploaded successfully to index '{self.index_name}'.")

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        logger.debug(f"Searching for {query} in index {self.index_name}")
        results = self.index_client.search(query=query, top=top_k)
        logger.debug(f"Results: {results}")
        return results
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        logger.debug(f"Retrieving for query: {query}")

        search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)

        results = search_client.search(
            search_text=query,
            top=top_k,
            query_type="semantic",
            semantic_configuration_name="default-semantic-config"
        )

        hits = {}
        for result in results:
            chunk_id = result["id"]
            score = result["@search.rerankerScore"] if "@search.rerankerScore" in result else result["@search.score"]
            hits[chunk_id] = score

        logger.info(f"Retrieved {len(hits)} chunks")
        return hits