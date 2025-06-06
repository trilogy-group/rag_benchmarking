from typing import List, Dict, Any
import os
from pinecone import Pinecone, ServerlessSpec
import time
from tqdm import tqdm
from datastores._datastore import DataStore
from enum import Enum
from embeddings.embedding_models import PineconeNativeEmbeddingModel
from openai import AzureOpenAI
from embeddings.embedding_helper import EmbeddingHelper




class PineconeDatastore(DataStore):
    def __init__(self, index_name: str, text_embedding_model: str, openai_model: str, namespace:str, agent_name: str="pinecone-agent",):
        self.index_name = index_name
        self.agent_name = agent_name
        self.text_embedding_model = text_embedding_model
        self.openai_model = openai_model
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.namespace = namespace
        self.embedding_helper = EmbeddingHelper(self.text_embedding_model)
        self.vector_size = self.embedding_helper.get_embedding_size()  
        print(f"🔍 Vector size: {self.vector_size}")
        self.create_index()



    def is_native_embedding_model(self, model_name: str) -> bool:
        print(f"🔍 Checking if {model_name} is a native embedding model")
        is_native = model_name in [model.value for model in PineconeNativeEmbeddingModel]
        print(f"🔍 {model_name} is a native embedding model: {is_native}")
        return is_native
    
        
    def create_index_custom_model(self):
        print(f"✅ Creating custom model index: {self.index_name}")
        
        if not self.pinecone_client.has_index(self.index_name):
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.vector_size,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        print(f"✅ Index created: {self.index_name}")

    def create_index_native_model(self):
        print(f"✅ Creating index: {self.index_name}")
        if not self.pinecone_client.has_index(self.index_name):
            self.pinecone_client.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": self.text_embedding_model,
                    "field_map":{
                        "text": "content"
                    }
                }
            )
            print(f"✅ Index created: {self.index_name}")


    def create_index(self):
        if self.is_native_embedding_model(self.text_embedding_model):
            print(f"🔍 Creating index for native embedding model: {self.text_embedding_model}")
            self.create_index_native_model()
        else:
            print(f"🔍 Creating index for custom embedding model: {self.text_embedding_model}")
            self.create_index_custom_model()

    
    def prepare_pinecone_records(self, corpus: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
        return [
            {
                "id": doc["id"],
                "values": embedding,
                "metadata": {
                    "text": doc["content"],
                    **{k: v for k, v in doc.items() if k not in ["id", "content"]}
                }
            }
            for doc, embedding in zip(corpus, embeddings)
        ]


    def index_corpus_pinecone_native_embedding(self, corpus: List[Dict[str, Any]]):
        if not corpus:
            print("⚠️ Empty corpus provided. Skipping indexing.")
            return
        
        dense_index = self.pinecone_client.Index(self.index_name)

        batch_size = 90
        for i in tqdm(range(0, len(corpus), batch_size), desc="Indexing batches"):
            try:
                batch = corpus[i:i+batch_size]
                dense_index.upsert_records(
                    records=batch,
                    namespace=self.namespace
                )
            except Exception as e:
                print(f"Error indexing batch {i}: {e}")
                time.sleep(10)
        print(f"✅ Indexed {len(corpus)} documents")
        time.sleep(10)

        # View stats for the index
        stats = dense_index.describe_index_stats()
        print(f"✅ Index stats: {stats}")

    def index_corpus(self, corpus: List[Dict[str, Any]]):
        if self.is_native_embedding_model(self.text_embedding_model):
            self.index_corpus_pinecone_native_embedding(corpus)
        else:
            self.index_corpus_custom_embedding(corpus)

    def index_corpus_custom_embedding(self, corpus: List[Dict[str, Any]]):
        if not corpus:
            print("⚠️ Empty corpus provided. Skipping indexing.")
            return

        dense_index = self.pinecone_client.Index(self.index_name)
        batch_size = 50

        for i in tqdm(range(0, len(corpus), batch_size), desc="📦 Indexing batches"):
            batch = corpus[i:i + batch_size]
            texts = [doc.get("content", "").strip() for doc in batch]
            texts = [t for t in texts if t]

            if not texts:
                print(f"⚠️ Skipping empty/invalid batch at index {i}")
                continue

            try:
                embeddings = self.embedding_helper.create_embeddings(batch)
                records = self.prepare_pinecone_records(batch, embeddings)

                dense_index.upsert(vectors=records, namespace=self.namespace)
                # time.sleep(10)

            except Exception as e:
                print(f"❌ Error processing batch {i}-{i + len(batch)}: {e}")
                time.sleep(5)

        print(f"✅ Successfully indexed {len(corpus)} documents")
        
    
    
    def retrieve(self, query: str, top_k: int = 10):
        pass