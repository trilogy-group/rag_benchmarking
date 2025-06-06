from embeddings.openai_embedding import OpenAIEmbedding
from embeddings.embedding_models import OpenAIEmbeddingModel, CohereEmbeddingModel, GeminiEmbeddingModel
from embeddings.gemini_embedding import GeminiEmbedding
from embeddings.cohere_embedding import CohereEmbedding
from typing import List, Dict, Any

class EmbeddingHelper:
    def __init__(self, model_name: str):
        if model_name in [e.value for e in OpenAIEmbeddingModel]:
            self.embedding_model = OpenAIEmbedding(model_name)
        elif model_name in [e.value for e in CohereEmbeddingModel]:
            self.embedding_model = CohereEmbedding(model_name)
        elif model_name in [e.value for e in GeminiEmbeddingModel]:
            self.embedding_model = GeminiEmbedding(model_name)
        else:
            raise ValueError(f"Invalid embedding model: {model_name}")
        
    
    def get_embedding_size(self) -> int:
        return self.embedding_model.get_embedding_size()
        

    def create_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        return self.embedding_model.create_embeddings(docs)

    