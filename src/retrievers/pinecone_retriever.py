from retrievers._retriever import Retriever
from typing import List, Tuple
import os
from pinecone import Pinecone
from embedding_models import PineconeNativeEmbeddingModel
import openai

class PineconeRetriever(Retriever):
    def __init__(self, index_name: str, agent_name: str, namespace: str, text_embedding_model: str):
        self.index_name = index_name
        self.agent_name = agent_name
        self.namespace = namespace
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.text_embedding_model = text_embedding_model

    def is_native_embedding_model(self, model_name: str) -> bool:
        return model_name in PineconeNativeEmbeddingModel.__members__


    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        dense_index = self.pinecone_client.Index(self.index_name)

        response = dense_index.search(
            namespace=self.namespace,
            query={
                "top_k": top_k,
                "inputs": {
                    "text": query
                }
            },
        )

        print(f"Pinecone results: {len(response['result']['hits'])}")

        hits = {}
        for match in response['result']["hits"]:
            doc_id = match["_id"]
            score = match["_score"]
            hits[doc_id] = score

        print(f"Retrieved {len(hits)} chunks from Pinecone")
        return hits
    
    def retrieve_old(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        dense_index = self.pinecone_client.Index(self.index_name)

        if self.is_native_embedding_model(self.text_embedding_model):
            response = dense_index.search(
                namespace=self.namespace,
                query={
                    "top_k": top_k,
                    "inputs": {
                        "text": query
                    }
                }
            )
            matches = response["result"]["hits"]
        else:
            embedding_response = openai.embeddings.create(
                input=[query],
                model=self.text_embedding_model
            )
            query_vector = embedding_response.data[0].embedding
            response = dense_index.search(
                namespace=self.namespace,
                vector=query_vector,
                top_k=top_k
            )
            matches = response["matches"]

        hits = [(match.get("id") or match.get("_id"), match.get("score") or match.get("_score")) for match in matches]
        print(f"🔍 Retrieved {len(hits)} chunks from Pinecone")
        return hits
