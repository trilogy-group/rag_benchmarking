from embeddings._embedding import Embedding
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
from sklearn.preprocessing import normalize
import torch
import numpy as np

class StellaEmbedding(Embedding):
    def __init__(self, model: str = "dunzhang/stella_en_1.5B_v5"):
        self.model_name = model
        print(f"Stella model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).eval()

    def get_embedding_size(self) -> int:
        return 1024  # Default output dimension for Stella v5

    def create_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc["content"] for doc in docs]

        print(f"📡 Generating embeddings for {len(texts)} documents using Stella model: {self.model_name}")
        print(f"Texts: {texts[:1]}")

        embeddings = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            norm_embedding = normalize([mean_embedding])[0]
            embeddings.append(norm_embedding.tolist())

        print(f"Successfully generated {len(embeddings)} embedding vectors.")

        if embeddings:
            print(f"The dimension of each embedding vector is: {len(embeddings[0])}")

        return embeddings
