from typing import List, Tuple, Dict, Any
import os
import json
import jsonlines
import logging

logger = logging.getLogger(__name__)

class DataStore:
    def index_corpus(self, dataset_name: str, documents: List[str], ids: List[str] = None) -> None:
        """
        Indexes a list of documents. Optionally takes document IDs.
        """
        raise NotImplementedError
    
    
    # def load_precomputed_embeddings(self, file_path: str) -> Dict[str, Dict[str, Any]]:
    #     if not os.path.exists(file_path):
    #         print(f"⚠️ No precomputed embeddings at: {file_path}")
    #         return {}

    #     print(f"📥 Loading precomputed embeddings from {file_path}")
    #     id_to_record = {}
    #     with open(file_path, "r") as f:
    #         for line in f:
    #             try:
    #                 record = json.loads(line.strip())
    #                 if all(k in record for k in ("id", "text", "embedding", "metadata")):
    #                     id_to_record[record["id"]] = record
    #                 else:
    #                     print(f"⚠️ Skipping malformed record: {record}")
    #             except json.JSONDecodeError as e:
    #                 print(f"❌ Error parsing line: {e}")
    #     return id_to_record

    def load_precomputed_embeddings(self, file_path: str) -> Dict[str, Dict[str, Any]]:    
        if not os.path.exists(file_path):
            logger.warning(f"No precomputed embeddings at: {file_path}")
            return {}

        logger.info(f"Loading precomputed embeddings from {file_path}")
        id_to_record = {}
        bad_lines = 0
        total = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                total += 1
                try:
                    record = json.loads(line.strip())
                    if all(k in record for k in ("id", "text", "embedding", "metadata")):
                        id_to_record[record["id"]] = record
                    else:
                        logger.warning(f"Skipping line {i}: missing required fields")
                        bad_lines += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Skipping line {i}: invalid JSON ({e})")
                    bad_lines += 1

        logger.info(f"Loaded {len(id_to_record)} valid records")
        if bad_lines:
            logger.warning(f"Skipped {bad_lines} malformed lines out of {total}")

        return id_to_record
    
    def append_records_to_file(
        self,
        docs: List[Dict[str, Any]],
        embeddings: List[List[float]],
        file_path: str
    ):
        """
        Appends embedded documents to a JSONL file in a neutral format.
        Each line is a JSON object with {id, text, metadata, embedding}.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "a") as f:
            for doc, vector in zip(docs, embeddings):
                json_record = {
                    "id": doc["id"],
                    "text": doc.get("content", ""),
                    "metadata": {k: v for k, v in doc.items() if k not in ["id", "content"]},
                    "embedding": vector,
                }
                f.write(json.dumps(json_record) + "\n")
