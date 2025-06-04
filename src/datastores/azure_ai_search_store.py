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


class AzureAISearchStore(DataStore):
    def __init__(self, index_name: str, agent_name: str, openai_model: str, text_embedding_model: str):
        self.index_name = index_name
        self.agent_name = agent_name
        self.openai_model = openai_model

        print(f"Index Name: {self.index_name}")
        print(f"Agent Name: {self.agent_name}")
        

        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.embedding_model = text_embedding_model
        self.embedding_deployment = text_embedding_model
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.agent_client = KnowledgeAgentRetrievalClient(endpoint=self.endpoint, agent_name=self.agent_name, credential=self.credential)
        print(f"Endpoint: {self.endpoint}")
        # print(f"Credential: {self.credential}")
        # print(f"OpenAI API Key: {self.azure_openai_api_key}")
        print(f"OpenAI Model: {self.openai_model}")
        print(f"Embedding Model: {self.embedding_model}")
        print(f"Embedding Deployment: {self.embedding_deployment}")
        print(f"OpenAI Endpoint: {self.azure_openai_endpoint}")
        self.setup_index()
        # self.setup_agent()

    def get_embedding_dimensions(self, model_name: str) -> int:
        return {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }.get(model_name, 1536)  # default to 1536 if unknown

    def setup_index(self):
        print(f"Setting up index {self.index_name}")

        index = SearchIndex(
            name=self.index_name,
            fields=[
                SearchField(name="id", type="Edm.String", key=True, filterable=True),
                SearchField(name="content", type="Edm.String"),
                SearchField(name="embedding", type="Collection(Edm.Single)", stored=False,            
                vector_search_dimensions=self.get_embedding_dimensions(self.embedding_model), 
                vector_search_profile_name="profile"),
            ],
            vector_search=VectorSearch(
                profiles=[VectorSearchProfile(name="profile", algorithm_configuration_name="alg", vectorizer_name="vec")],
                algorithms=[HnswAlgorithmConfiguration(name="alg")],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        vectorizer_name="vec",
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=self.azure_openai_endpoint,
                            api_key=self.azure_openai_api_key,
                            model_name=self.embedding_model,
                            deployment_name=self.embedding_deployment
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
        print(f"Index {index} created")
    
    def setup_agent(self):
        print(f"Setting up agent {self.agent_name}")
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
            target_indexes=[KnowledgeAgentTargetIndex(index_name=self.index_name)]
        )
       
        self.index_client.create_or_update_agent(agent)
        print(f"Agent {self.agent_name} created")

    def index_corpus_simple(self, documents: List[str], ids: List[str] = None) -> None:
        print(f"Indexing {len(documents)} documents to index {self.index_name}")
        with SearchIndexingBufferedSender(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential) as sender:
            print(f"Uploading {len(documents)} documents to index {self.index_name}")
            sender.upload_documents(documents=documents)
        print(f"Documents uploaded to index {self.index_name}")
    
    def count_documents(self) -> int:
        search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)
        results = search_client.search(search_text="*")
        return len(list(results))

    def index_corpus(self, documents: List[dict], ids: List[str] = None, batch_size: int = 100) -> None:
        total = len(documents)
        if total == 0:
            print("No documents to index.")
            return

        print(f"Indexing {total} documents to index '{self.index_name}' in batches of {batch_size}.")

        try:
            with SearchIndexingBufferedSender(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential
            ) as sender:

                for i in tqdm(range(0, total, batch_size), desc="Uploading batches"):
                    batch = documents[i:i + batch_size]
                    print(f"Uploading batch {i // batch_size + 1}: type={type(batch)}")
                    sender.upload_documents(batch)

            print(f"All {total} documents uploaded successfully to index '{self.index_name}'.")

        except Exception as e:
            print(f"Failed to index documents: {e}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        print(f"Searching for {query} in index {self.index_name}")
        results = self.index_client.search(query=query, top=top_k)
        print(f"Results: {results}")
        return results
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        print(f"Retrieving for query: {query}")

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

        print(f"Retrieved {len(hits)} chunks")
        return hits