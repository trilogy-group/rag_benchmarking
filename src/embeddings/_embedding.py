from typing import List, Dict, Any

class Embedding:
    def __init__(self, model: str):
        self.model = model

    def create_embeddings(self, corpus: List[Dict[str, Any]]) -> List[List[float]]:
        pass

