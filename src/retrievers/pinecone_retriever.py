from retrievers._retriever import Retriever
from typing import List, Tuple
import os
from pinecone import Pinecone
from embeddings.embedding_models import PineconeNativeEmbeddingModel
from embeddings.embedding_helper import EmbeddingHelper
import openai
import logging

logger = logging.getLogger(__name__)

class PineconeRetriever(Retriever):
    def __init__(self, index_name: str, agent_name: str, namespace: str, text_embedding_model: str):
        self.index_name = index_name
        self.agent_name = agent_name
        self.namespace = namespace
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.text_embedding_model = text_embedding_model
        self.embedding_helper = EmbeddingHelper(self.text_embedding_model)

    def is_native_embedding_model(self, model_name: str) -> bool:
        logger.debug(f"Checking if {model_name} is a native embedding model")
        is_native = model_name in [model.value for model in PineconeNativeEmbeddingModel]
        logger.debug(f"{model_name} is a native embedding model: {is_native}")
        return is_native
    

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.is_native_embedding_model(self.text_embedding_model):
            return self.retrieve_from_native_pinecone_index(query, top_k)
        else:
            return self.retrieve_from_custom_index(query, top_k)

    def retrieve_from_native_pinecone_index(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
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
        
        logger.debug(f"Pinecone results: {len(response['result']['hits'])}")

        hits = {}
        for match in response['result']["hits"]:
            doc_id = match["_id"]
            score = match["_score"]
            hits[doc_id] = score

        logger.info(f"Retrieved {len(hits)} chunks from Pinecone")
        return hits
    
    def retrieve_from_custom_index(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        dense_index = self.pinecone_client.Index(self.index_name)
        query_vector = self.embedding_helper.create_embeddings([{"content": query}])[0]
        response = dense_index.search(
            namespace=self.namespace,
            query={
                "vector": {
                    "values": query_vector
                },
                "top_k": top_k
            }
        )
        logger.debug(f"Pinecone results: {len(response['result']['hits'])}")

        hits = {}
        for match in response['result']["hits"]:
            doc_id = match["_id"]
            score = match["_score"]
            hits[doc_id] = score

        logger.info(f"Retrieved {len(hits)} chunks from Pinecone")
        return hits