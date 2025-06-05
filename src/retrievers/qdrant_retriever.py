from retrievers._retriever import Retriever
from typing import List, Dict
import os
from pinecone import Pinecone
from embedding_models import PineconeNativeEmbeddingModel
from qdrant_client import QdrantClient
from openai import OpenAI

class QdrantRetriever(Retriever):
    def __init__(self, index_name: str, agent_name: str, namespace: str, text_embedding_model: str):
        self.index_name = index_name
        self.agent_name = agent_name
        self.namespace = namespace
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.text_embedding_model = text_embedding_model

        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_CLOUD_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        # OpenAI setup
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def retrieve(self, query: str, top_k: int = 10) -> Dict[str, float]:
        try:
            print(f"🔍 Embedding query: '{query}' using model '{self.text_embedding_model}'")
            response = self.openai_client.embeddings.create(
                input=[query],
                model=self.text_embedding_model
            )

            query_vector = response.data[0].embedding

            print(f"📡 Performing Qdrant search in collection: {self.index_name}")
            # print(f"Query vector: {query_vector}")

            results = self.qdrant_client.search(
                collection_name=self.index_name,
                query_vector=query_vector,
                limit=top_k
            )

            # print(f"Results: {results}")
            print(f"Qdrant returned {len(results)} hits")
            hits = {hit.payload["doc_id"] or hit.id: hit.score for hit in results}

            print(f"Retrieved {len(hits)} chunks from Qdrant")
            return hits

        except Exception as e:
            print(f"❌ Qdrant retrieval failed: {e}")
            return {}
    


    

    