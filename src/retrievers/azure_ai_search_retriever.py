from retrievers._retriever import Retriever
from typing import List, Tuple
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizerParameters,
    KnowledgeAgent,
    KnowledgeAgentTargetIndex,
    KnowledgeAgentAzureOpenAIModel
)
from azure.search.documents.indexes import SearchIndexClient
# Agentic retrieval components (used in retrieve_agentic)
from azure.search.documents.agent.models import (
    KnowledgeAgentIndexParams,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentRetrievalRequest
)


import os

class AzureAISearchRetriever(Retriever):
    def __init__(self, index_name: str, agent_name: str, namespace: str, text_embedding_model: str, openai_model: str="gpt-4o"):
        self.index_name = index_name
        self.agent_name = agent_name
        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        self.namespace = namespace
        self.text_embedding_model = text_embedding_model
        self.openai_model = openai_model
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self.agent_client = KnowledgeAgentRetrievalClient(endpoint=self.endpoint, agent_name=self.agent_name, credential=self.credential)

        
    def retrieve_semantic(self, query: str, top_k: int) -> List[Tuple[str, float]]:
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
    
    
    def retrieve_agentic(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        print(f"Retrieving agentic response from agent {self.agent_name}")

        instructions = """
        A Q&A agent that can answer questions about the Earth at night.
        Sources have a JSON format with a ref_id that must be cited in the answer.
        If you do not have the answer, respond with "I don't know".
        """

        # Construct structured KnowledgeAgentMessage objects
        messages = [
            KnowledgeAgentMessage(
                role="assistant",
                content=[KnowledgeAgentMessageTextContent(text=instructions)]
            ),
            KnowledgeAgentMessage(
                role="user",
                content=[KnowledgeAgentMessageTextContent(text=query)]
            )
        ]

        # Define index params with reranker threshold (optional: tweak as needed)
        index_params = [
            KnowledgeAgentIndexParams(
                index_name=self.index_name,
                reranker_threshold=2.5  # or whatever makes sense for your eval
            )
        ]

        try:
            retrieval_result = self.agent_client.knowledge_retrieval.retrieve(
                retrieval_request=KnowledgeAgentRetrievalRequest(
                    messages=messages,
                    target_index_params=index_params
                )
            )       
        except Exception as e:
            print(f"Error retrieving agentic response: {e}")
            return {}



        # Extract citations / hits
        hits = {}
        for citation in retrieval_result.citations:
            chunk_id = citation.chunk_id
            score = citation.score if hasattr(citation, "score") else 1.0
            hits[chunk_id] = score

        print(f"Agentic retrieval returned {len(hits)} citations")
        return hits
    
    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        return self.retrieve_agentic(query, top_k)
        
