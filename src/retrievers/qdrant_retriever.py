from retrievers._retriever import Retriever
from typing import List, Dict
import os
from pinecone import Pinecone
from embeddings.embedding_models import PineconeNativeEmbeddingModel
from qdrant_client import QdrantClient
from openai import OpenAI
from embeddings.embedding_helper import EmbeddingHelper

class QdrantRetriever(Retriever):
    def __init__(self, index_name: str, agent_name: str, namespace: str, text_embedding_model: str):
        self.index_name = index_name
        self.agent_name = agent_name
        self.namespace = namespace
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.text_embedding_model = text_embedding_model

        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_CLOUD_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30.0
        )

        # OpenAI setup
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def retrieve(self, query: str, top_k: int = 10) -> Dict[str, float]:
        try:

            embedding_helper = EmbeddingHelper(self.text_embedding_model)
            query_vector = embedding_helper.create_embeddings([{"content": query}])[0]

            results = self.qdrant_client.search(
                collection_name=self.index_name,
                query_vector=query_vector,
                limit=top_k
            )

            hits = {hit.payload["doc_id"] or hit.id: hit.score for hit in results}

            return hits

        except Exception as e:
            print(f"❌ Qdrant retrieval failed: {e}")
            return {}
    


    

    