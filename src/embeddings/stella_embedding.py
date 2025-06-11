from embeddings._embedding import Embedding
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
from sklearn.preprocessing import normalize
import torch
import numpy as np

class StellaEmbedding(Embedding):
    def __init__(self, model: str):
        self.model_name = model
        print(f"🧠 Loading Stella model: {self.model_name}")

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Using CUDA")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🍎 Using Apple MPS")
        else:
            self.device = torch.device("cpu")
            print("🖥️ Using CPU")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Model (no accelerate needed)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=False
        ).to(device=self.device).eval()

    def get_embedding_size(self) -> int:
        return 1536  # Adjust if needed for your model

    def create_embeddings(self, docs: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc["content"] for doc in docs]

        print(f"📡 Generating embeddings for {len(texts)} documents using {self.model_name}")
        embeddings = []

        for text in texts:
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                norm_embedding = normalize([mean_embedding])[0]
                embeddings.append(norm_embedding.tolist())
            except Exception as e:
                print(f"Error generating embedding for text: {text}")
                print(f"Error: {e}")
                continue

        print(f"✅ Created {len(embeddings)} embedding vectors.")
        return embeddings
