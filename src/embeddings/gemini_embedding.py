from embeddings._embedding import Embedding
from google import genai
import os
from typing import List, Dict, Any

class GeminiEmbedding(Embedding):
    def __init__(self, model: str):
        self.model = model
        print(f"Gemini model: {self.model}")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def get_embedding_size(self) -> int:
        return 3072
    
    def create_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc["content"] for doc in docs]

        print(f"📡 Generating embeddings for {len(texts)} documents using Gemini model: {self.model}")
        print(f"Texts: {texts[:1]}")

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts
        )

        embeddings = [embedding.values for embedding in response.embeddings]

        print(f"Successfully generated {len(embeddings)} embedding vectors.")
        
        # Check if vectors were generated before accessing an element
        if embeddings:
            print(f"The dimension of each embedding vector is: {len(embeddings[0])}")

        return embeddings



