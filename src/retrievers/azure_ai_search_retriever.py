from retrievers._retriever import Retriever
from typing import List, Tuple
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os

class AzureAISearchRetriever(Retriever):
    def __init__(self, index_name: str, agent_name: str):
        self.index_name = index_name
        self.agent_name = agent_name
        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))


    # def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    #     print(f"Retrieving for query: {query}")

    #     search_client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)

    #     results = search_client.search(
    #         search_text=query,
    #         top=top_k,
    #         query_type="semantic",
    #         semantic_configuration_name="default-semantic-config"
    #     )

    #     hits = {}
    #     for result in results:
    #         chunk_id = result["id"]
    #         score = result["@search.rerankerScore"] if "@search.rerankerScore" in result else result["@search.score"]
    #         hits[chunk_id] = score

    #     print(f"Retrieved {len(hits)} chunks")
    #     return hits

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
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
