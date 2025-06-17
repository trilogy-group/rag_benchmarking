from embeddings._embedding import Embedding
from google import genai
from google.genai.types import EmbedContentConfig
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class GeminiEmbedding(Embedding):
    def __init__(self, model: str):
        logger.info(f"Init GeminiEmbedding model: {model}")
        self.model = model
        logger.debug(f"Gemini model: {self.model}")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.vertex_client = genai.Client()

    def get_embedding_size(self) -> int:
        return 3072
    
    def create_vertex_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(docs)} documents using Gemini vertex model: {self.model}")
        texts = [doc["content"] for doc in docs]
        embeddings = [None] * len(texts)  # Preallocate to preserve order

        def embed_text(index: int, text: str):
            try:
                response = self.vertex_client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=EmbedContentConfig(
                        output_dimensionality=self.get_embedding_size(),
                    ),
                )
                vector = response.embeddings[0].values
                logger.debug(f"Embedding {index + 1}/{len(texts)}: dimension {len(vector)}")
                return index, vector
            except Exception as e:
                logger.error(f"Failed to embed document {index + 1}: {e}")
                return index, []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(embed_text, i, text): i for i, text in enumerate(texts)}
            for future in as_completed(futures):
                index, vector = future.result()
                embeddings[index] = vector

        # Filter out failed embeddings (if any)
        successful_embeddings = [vec for vec in embeddings if vec]

        logger.info(f"Successfully generated {len(successful_embeddings)} embedding vectors.")
        if successful_embeddings:
            logger.debug(f"Each embedding vector dimension: {len(successful_embeddings[0])}")

        return embeddings  
    
    def create_legacy_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc["content"] for doc in docs]

        logger.info(f"Generating embeddings for {len(texts)} documents using Gemini model: {self.model}")
        logger.debug(f"Texts: {texts[:1]}")

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts
        )

        embeddings = [embedding.values for embedding in response.embeddings]
        logger.info(f"Successfully generated {len(embeddings)} embedding vectors.")
        
        if embeddings:
            logger.debug(f"The dimension of each embedding vector is: {len(embeddings[0])}")

        return embeddings


    def create_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        if self.model == "gemini-embedding-001":
            return self.create_vertex_embeddings(docs)
        else:
            return self.create_legacy_embeddings(docs)