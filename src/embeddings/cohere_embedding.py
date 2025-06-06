from typing import List, Dict, Any
from openai import OpenAI
import os
from embeddings._embedding import Embedding
from cohere import ClientV2

class CohereEmbedding(Embedding):
    def __init__(self, model: str):
        self.model = model
        self.cohere_client = ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    def get_embedding_size(self) -> int:
        return 1536

    def create_embeddings(self, corpus: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc["content"] for doc in corpus]
        print(f"📡 Generating cohere embeddings for {len(texts)} documents using model: {self.model}")
        # print(f"Texts: {texts[:5]}")

        response = self.cohere_client.embed(
            texts=texts, 
            model=self.model,
            input_type="search_query",
            embedding_types=["float"]
        )
        print(f"Response: {len(response.embeddings.float[0])}")
        return response.embeddings.float