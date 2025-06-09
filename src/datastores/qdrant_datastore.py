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
        
    def index_corpus(self, corpus: List[Dict[str, Any]]):
        if not corpus:
            print("⚠️ Empty corpus provided. Skipping indexing.")
            return

        print(f"📥 Indexing {len(corpus)} documents to Qdrant collection: {self.collection_name}")
        batch_size = 50
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]

        def process_batch(batch_index: int, batch: List[Dict[str, Any]]):
            texts = [doc.get("content", "").strip() for doc in batch]
            texts = [t for t in texts if t]

            if not texts:
                print(f"⚠️ Skipping empty/invalid batch at index {batch_index}")
                return False

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    embeddings = self.embedding_helper.create_embeddings(batch)
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
                        for doc, embedding in zip(batch, embeddings)
                    ]
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    return True
                except Exception as e:
                    print(f"❌ Batch {batch_index} attempt {attempt} failed: {e}")
                    time.sleep(5)
            print(f"❌ Batch {batch_index} failed after {max_retries} attempts.")
            return False

        successes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(process_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            with tqdm(total=len(batches), desc="📦 Indexing batches") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    success = future.result()
                    if success:
                        successes += 1
                    pbar.update(1)

        print(f"✅ Indexed {successes}/{len(batches)} batches successfully ({successes * batch_size} documents estimated)")

    # def index_corpus(self, corpus: List[Dict[str, Any]]):
    #     if not corpus:
    #         print("⚠️ Empty corpus provided. Skipping indexing.")
    #         return

    #     print(f"📥 Indexing {len(corpus)} documents to Qdrant collection: {self.collection_name}")

    #     batch_size = 50
    #     for i in tqdm(range(0, len(corpus), batch_size), desc="📦 Indexing batches"):
    #         batch = corpus[i:i + batch_size]
    #         texts = [doc.get("content", "").strip() for doc in batch]
    #         texts = [t for t in texts if t]

    #         print(f"Batch: {batch[0]} {len(batch)}")
    #         print(f"Texts: {texts[0]} {len(texts)}")

    #         if not texts:
    #             print(f"⚠️ Skipping empty/invalid batch at index {i}")
    #             continue

    #         try:
    #             embeddings = self.embedding_helper.create_embeddings(batch)
    #             points = [
    #                 qdrant_models.PointStruct(
    #                     id=str(uuid.UUID(doc["id"])) if self._is_valid_uuid(doc["id"]) else str(uuid.uuid4()),
    #                     vector=embedding,
    #                     payload={
    #                         "text": doc["content"],
    #                         "doc_id": doc["id"],
    #                         **{k: v for k, v in doc.items() if k not in ["id", "content"]}
    #                     }
    #                 )
    #                 for doc, embedding in zip(batch, embeddings)
    #             ]

    #             self.qdrant_client.upsert(
    #                 collection_name=self.collection_name,
    #                 points=points
    #             )
    #         except Exception as e:
    #             print(f"❌ Error processing batch {i}-{i + len(batch)}: {e}")
    #             time.sleep(5)

    #     print(f"✅ Successfully indexed {len(corpus)} documents")


    def retrieve(self, query: str, top_k: int = 10):
        # Retrieval not yet implemented
        pass


