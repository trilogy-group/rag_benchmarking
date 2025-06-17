from embeddings._embedding import Embedding
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
from sklearn.preprocessing import normalize
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModernBERTEmbedding(Embedding):
    def __init__(self, model: str):
        self.model_name = model
        logger.info(f"ModernBERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).eval()

    def get_embedding_size(self) -> int:
        return 1024  # ModernBERT Large typically uses 1024D

    def create_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc["content"] for doc in docs]

        logger.info(f"Generating embeddings for {len(texts)} documents using ModernBERT model: {self.model_name}")
        # print(f"Texts: {texts[:1]}")

        embeddings = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)

            mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            norm_embedding = normalize([mean_embedding])[0]  # Optional L2 normalization
            embeddings.append(norm_embedding.tolist())

        logger.info(f"Successfully generated {len(embeddings)} embedding vectors.")

        if embeddings:
            logger.debug(f"The dimension of each embedding vector is: {len(embeddings[0])}")

        return embeddings
