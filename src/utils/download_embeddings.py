import os
import json
from tqdm import tqdm
from pinecone import Pinecone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


# Config
INDEX_NAME = "pinecone-frame-gemini-001-gpt-4o-index"
NAMESPACE = "pinecone-frame-gemini-001-gpt-4o"
OUTPUT_FILE_PATH = "data/benchmark_datasets/frames/default/embeddings/gemini-embedding-001.jsonl"
IDS_FILE_PATH = "data/benchmark_datasets/frames/retrieval_format/corpus.json"
CORPUS_SIZE=200000
BATCH_SIZE=30

load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)


def convert_json_to_jsonl(input_path: str, output_path: str):
    """
    Converts a JSON file with top-level keys as IDs to JSONL format.
    Each line in the output will be a single JSON object that includes the original key as 'id'.
    
    Args:
        input_path (str): Path to the input .json file
        output_path (str): Path to the output .jsonl file
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for doc_id, content in data.items():
            # Add the key as an 'id' field in each entry
            content_with_id = {"id": doc_id, **content}
            outfile.write(json.dumps(content_with_id) + '\n')

    print(f"Converted {len(data)} records to JSONL and saved to {output_path}")

def load_ids_from_jsonl(path):
    convert_json_to_jsonl(path, path.replace(".json", ".jsonl"))
    path = path.replace(".json", ".jsonl")
    
    print(f"📂 Loading vector IDs from: {path}")
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    ids.append(obj["id"])
                elif "_id" in obj:
                    ids.append(obj["_id"])
                elif "doc_id" in obj:
                    ids.append(obj["doc_id"])
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON line: {e}")
    print(f"✅ Loaded {len(ids)} IDs")
    return ids

def download_all_ids(index, namespace, batch_size=100):
    print("🔍 Fetching all vector IDs...")
    id_list = []
    result = index.describe_index_stats(namespace=namespace)
    total_vectors = result.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
    print(f"✅ Total vectors in namespace '{namespace}': {total_vectors}")

    # Pinecone v3-style generator
    for page in index.list(namespace=namespace, limit=batch_size):
        id_list.append(page)
        print(f"🔍 Fetched {len(id_list)} IDs")

    print(f"📦 Total fetched IDs: {len(id_list)} {id_list[0]}")
    return id_list

def load_existing_ids(path):
    if not os.path.exists(path):
        return set()
    print(f"🔍 Reading existing IDs from {path}")
    existing_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    existing_ids.add(obj["id"])
            except json.JSONDecodeError:
                continue
    print(f"✅ Found {len(existing_ids)} already saved IDs")
    return existing_ids

def fetch_vectors_and_write(index, ids, namespace, output_path):
    print("📥 Downloading and writing vectors in batches (parallelized)...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def fetch_batch(batch_ids, i):
        try:
            print(f"➡️ [Thread] Fetching batch {i} to {i + len(batch_ids) - 1}...")
            res = index.fetch(ids=batch_ids, namespace=namespace)
            vectors = res.vectors

            records = []
            for vec_id, vec_data in vectors.items():
                embedding = vec_data.values
                metadata = dict(vec_data.metadata or {})
                text = metadata.pop("text", "")
                records.append({
                    "id": vec_id,
                    "text": text,
                    "metadata": metadata,
                    "embedding": embedding
                })
            return records
        except Exception as e:
            print(f"❌ Error in batch {i} to {i + len(batch_ids) - 1}: {e}")
            return []

    # Prepare all batches
    batches = [(ids[i:i + BATCH_SIZE], i) for i in range(0, len(ids), BATCH_SIZE)]

    with open(output_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(fetch_batch, batch_ids, i): (batch_ids, i) for batch_ids, i in batches}

            for future in tqdm(as_completed(futures), total=len(futures), desc="📦 Downloading"):
                records = future.result()
                for record in records:
                    json.dump(record, f)
                    f.write("\n")



if __name__ == "__main__":
    ids = load_ids_from_jsonl(IDS_FILE_PATH)
    ids = ids[:CORPUS_SIZE]

    # Load already-written IDs
    existing_ids = load_existing_ids(OUTPUT_FILE_PATH)
    
    # Filter only new IDs
    remaining_ids = [id_ for id_ in ids if id_ not in existing_ids]
    print(f"⚙️ Remaining IDs to fetch: {len(remaining_ids)} of {len(ids)} total")

    fetch_vectors_and_write(index, remaining_ids, NAMESPACE, OUTPUT_FILE_PATH)