from typing import List, Dict, Any
import os
import time
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from datastores._datastore import DataStore
from embeddings.embedding_helper import EmbeddingHelper
import uuid
import concurrent.futures
import json

class QdrantDatastore(DataStore):
    def __init__(
        self,
        text_embedding_model: str,
        openai_model: str,
        index_name: str,
        agent_name: str = "qdrant-agent",
        namespace: str = "default"
    ):
        self.collection_name = index_name
        self.text_embedding_model = text_embedding_model
        self.openai_model = openai_model
        self.agent_name = agent_name
        self.index_name = index_name
        # Qdrant setup
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_CLOUD_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        self.embedding_helper = EmbeddingHelper(self.text_embedding_model)
        self.vector_size = self.embedding_helper.get_embedding_size()  
        self.create_collection()

    def create_collection(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            print(f"✅ Creating Qdrant collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.vector_size,
                    distance=qdrant_models.Distance.COSINE
                )
            )
        else:
            print(f"ℹ️ Qdrant collection already exists: {self.collection_name}")
    
    def prepare_qdrant_records(self, corpus: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
        points = []
        for doc, embedding in zip(corpus, embeddings):
            print(f"Doc: {doc['id']}")
            print(f"Doc: {doc['content']}")
            points.append(
                qdrant_models.PointStruct(
                    id=doc["id"],
                    vector=embedding,
                    payload={
                        "text": doc["content"],
                        **{k: v for k, v in doc.items() if k not in ["id", "content"]}
                    }
                )
            )
        return points
        
    def _is_valid_uuid(self, value: str) -> bool:
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            return False
        
    def index_corpus(self, embeddings_file_path: str, corpus: List[Dict[str, Any]]):
        if not corpus:
            print("⚠️ Empty corpus provided. Skipping indexing.")
            return

        precomputed_map = self.load_precomputed_embeddings(embeddings_file_path)

        to_embed = []
        embedded_points = []

        for doc in corpus:
            doc_id = doc["id"]
            if doc_id in precomputed_map:
                rec = precomputed_map[doc_id]
                embedded_points.append(
                    qdrant_models.PointStruct(
                        id=str(uuid.UUID(doc_id)) if self._is_valid_uuid(doc_id) else str(uuid.uuid4()),
                        vector=rec["embedding"],
                        payload={
                            "text": rec["text"],
                            "doc_id": doc_id,
                            **rec.get("metadata", {})
                        }
                    )
                )
            else:
                to_embed.append(doc)

        print(f"✅ {len(embedded_points)} precomputed docs found")
        print(f"🧠 {len(to_embed)} docs require embedding computation")

        # Index precomputed in batches of 100 with tqdm
        precomputed_batch_size = 50
        precomputed_batches = [
            embedded_points[i:i + precomputed_batch_size]
            for i in range(0, len(embedded_points), precomputed_batch_size)
        ]

        if precomputed_batches:
            print("📦 Indexing precomputed embeddings...")
            for i, batch in enumerate(tqdm(precomputed_batches, desc="📥 Precomputed batches")):
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

        if not to_embed:
            return

        # Index new embeddings (computed)
        compute_batch_size = 50
        batches = [to_embed[i:i + compute_batch_size] for i in range(0, len(to_embed), compute_batch_size)]

        def process_batch(batch_index: int, batch: List[Dict[str, Any]]):
            valid_docs = [doc for doc in batch if doc.get("content", "").strip()]
            if not valid_docs:
                print(f"⚠️ Skipping empty batch {batch_index}")
                return False

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    embeddings = self.embedding_helper.create_embeddings(valid_docs)
                    self.append_records_to_file(valid_docs, embeddings, embeddings_file_path)
                    points = [
                        qdrant_models.PointStruct(
                            id=str(uuid.UUID(doc["id"])) if self._is_valid_uuid(doc["id"]) else str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "text": doc["content"],
                                "doc_id": doc["id"],
                                **{k: v for k, v in doc.items() if k not in ["id", "content"]}
                            }
                        )
                        for doc, embedding in zip(valid_docs, embeddings)
                    ]
                    self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                    return True
                except Exception as e:
                    print(f"❌ Batch {batch_index} attempt {attempt} failed: {e}")
                    time.sleep(5)
            print(f"❌ Batch {batch_index} failed after {max_retries} attempts.")
            return False

        successes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            with tqdm(total=len(batches), desc="📦 Indexing new embeddings") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    success = future.result()
                    if success:
                        successes += 1
                    pbar.update(1)

        print(f"✅ Indexed {successes}/{len(batches)} batches of new embeddings")
    

    def retrieve(self, query: str, top_k: int = 10):
        # Retrieval not yet implemented
        pass


