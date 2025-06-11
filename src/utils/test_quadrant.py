from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30.0  # add this explicitly
)

point = models.PointStruct(
    id=str(uuid.uuid4()),
    vector=[0.1] * 1536,  # adjust based on your embedding size
    payload={"text": "test point"}
)

try:
    collection_name = "test-collection"
    if not client.collection_exists(collection_name):
        print(f"🆕 Creating {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
    else:
        print(f"✅ Collection {collection_name} already exists")
    client.upsert(collection_name=collection_name, points=[point])
    print("✅ Point upserted successfully.")
except Exception as e:
    print(f"❌ Failed to upsert: {e}")