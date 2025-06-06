from typing import List, Dict, Any
from openai import OpenAI
import os
from embeddings._embedding import Embedding

class OpenAIEmbedding(Embedding):
    def __init__(self, model: str):
        self.model = model

        # OpenAI setup
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def get_embedding_size(self) -> int:
        return 3072

    def create_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc["content"] for doc in docs]

        print(f"📡 Generating embeddings for {len(texts)} documents using OpenAI model: {self.model}")
        # print(f"Texts: {texts[:5]}")

        response = self.openai_client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]