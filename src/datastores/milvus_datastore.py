from typing import List, Dict, Any
import os
import time
import uuid
from tqdm import tqdm
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)
from datastores._datastore import DataStore
from embeddings.embedding_helper import EmbeddingHelper
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

class MilvusDatastore(DataStore):
    def __init__(
        self,
        text_embedding_model: str,
        openai_model: str,
        index_name: str,
        agent_name: str = "milvus-agent",
        namespace: str = "default"
    ):
        self.collection_name = index_name
        self.text_embedding_model = text_embedding_model
        self.openai_model = openai_model
        self.agent_name = agent_name
        self.index_name = index_name

        self.embedding_helper = EmbeddingHelper(self.text_embedding_model)
        self.vector_size = self.embedding_helper.get_embedding_size()

        connections.connect(
            alias="default",
            uri=os.getenv("MILVUS_ENDPOINT"),
            token=f"{os.getenv('MILVUS_USERNAME')}:{os.getenv('MILVUS_PASSWORD')}"
        )

        self.create_collection()

    def create_collection(self):
        if not utility.has_collection(self.collection_name):
            logger.info(f"Creating Milvus collection: {self.collection_name}")
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=36),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_size),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            ]
            schema = CollectionSchema(fields=fields, description="Text embedding collection")
            Collection(name=self.collection_name, schema=schema)
        else:
            logger.info(f"Milvus collection already exists: {self.collection_name}")

    def index_corpus(self, corpus: List[Dict[str, Any]]):
        if not corpus:
            logger.warning("Empty corpus provided. Skipping indexing.")
            return

        logger.info(f"Indexing {len(corpus)} documents to Milvus collection: {self.collection_name}")
        collection = Collection(self.collection_name)

        batch_size = 50
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]

        def process_batch(batch_index: int, batch: List[Dict[str, Any]]):
            texts = [doc.get("content", "").strip() for doc in batch if doc.get("content")]
            if not texts:
                logger.warning(f"Skipping empty/invalid batch at index {batch_index}")
                return False

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    embeddings = self.embedding_helper.create_embeddings(batch)
                    ids = [
                        doc["id"]
                        for doc in batch
                    ]
                    contents = [doc["content"] for doc in batch]

                    insert_data = [ids, embeddings, contents]
                    collection.insert(insert_data)
                    return True
                except Exception as e:
                    logger.warning(f"Batch {batch_index} attempt {attempt} failed: {e}")
                    time.sleep(5)
            logger.error(f"Batch {batch_index} failed after {max_retries} attempts.")
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

        logger.info(f"Indexed {successes}/{len(batches)} batches successfully ({successes * batch_size} documents estimated)")

    def retrieve(self, query: str, top_k: int = 10):
        # Retrieval logic can be implemented using Milvus's `search` API.
        pass
