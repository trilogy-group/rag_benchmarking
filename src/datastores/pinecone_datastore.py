from typing import List, Dict, Any
import os
from pinecone import Pinecone, ServerlessSpec
import time
from tqdm import tqdm
from datastores._datastore import DataStore
from enum import Enum
from embedding_models import PineconeNativeEmbeddingModel
from openai import AzureOpenAI




class PineconeDatastore(DataStore):
    def __init__(self, index_name: str, text_embedding_model: str, openai_model: str, namespace:str, agent_name: str="pinecone-agent",):
        self.index_name = index_name
        self.agent_name = agent_name
        self.text_embedding_model = text_embedding_model
        self.openai_model = openai_model
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.namespace = namespace
        self.create_index()
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-04-14",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

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
                dimension=3072,
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

    def create_embeddings(self, corpus: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc["content"] for doc in corpus]
        print(f"📡 Generating embeddings for {len(texts)} documents with model: {self.text_embedding_model}")
        res = self.openai_client.embeddings.create(input=texts, model=self.text_embedding_model)
        return [r.embedding for r in res.data]
    
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


    def index_corpus(self, corpus: List[Dict[str, Any]]):
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
    
    def index_corpus_old(self, corpus: List[Dict[str, Any]]):
        if not corpus:
            print("⚠️ Empty corpus provided. Skipping indexing.")
            return

        index = self.pinecone_client.Index(self.index_name)

        if not self.is_native_embedding_model(self.text_embedding_model):
            embeddings = self.create_embeddings(corpus)
            records = self.prepare_pinecone_records(corpus, embeddings)
        else:
            records = [
                {
                    "id": doc["id"],
                    "metadata": {
                        "text": doc["content"],
                        **{k: v for k, v in doc.items() if k not in ["id", "content"]}
                    }
                }
                for doc in corpus
            ]

        batch_size = 90

        for i in tqdm(range(0, len(records), batch_size), desc="📥 Indexing batches"):
            batch = records[i:i + batch_size]
            try:
                index.upsert(records=batch, namespace=self.namespace)
            except Exception as e:
                print(f"❌ Error indexing batch {i}: {e}")
                time.sleep(10)

        print(f"✅ Successfully indexed {len(records)} documents")
        time.sleep(5)
        stats = index.describe_index_stats()
        print(f"📊 Index stats: {stats}")

    def retrieve(self, query: str, top_k: int = 10):
        pass