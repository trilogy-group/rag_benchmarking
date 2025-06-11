from typing import List, Dict, Any
import os
import time
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from datastores._datastore import DataStore
from embeddings.embedding_models import IntfloatEmbeddingModel, GeminiEmbeddingModel
import uuid
import concurrent.futures
from vertexai.preview import rag
from google.cloud import storage
import math

class VertexAiDatastore(DataStore):
    def __init__(
        self,
        text_embedding_model: str,
        openai_model: str,
        index_name: str,
        agent_name: str = "qdrant-agent",
        namespace: str = "default"
    ):
        self.corpus_name = index_name
        self.text_embedding_model = text_embedding_model
        self.agent_name = agent_name
        self.index_name = index_name
        self.gcs_bucket_name = os.getenv("VERTEX_CORPUS_BUCKET_NAME")
        self.bucket_prefix = index_name

        self.publisher_name = self.get_publisher_name(self.text_embedding_model)
        self.embedding_model_config = rag.EmbeddingModelConfig(
            publisher_model=f"publishers/{self.publisher_name}/models/{self.text_embedding_model}"
        )

        self.create_corpus()

    def get_publisher_name(self, model_name: str) -> str:
        if model_name in [e.value for e in GeminiEmbeddingModel]:
            return "google"
        elif model_name is IntfloatEmbeddingModel.ML_E5_LARGE.value:
            return "intfloat"
        # elif model_name in [e.value for e in CohereEmbeddingModel]:
        #     return "cohere"
        # elif model_name in [e.value for e in GeminiEmbeddingModel]:
        #     return "google"
        # elif model_name in [e.value for e in StellaEmbeddingModel]:
        #     return "stella"
        else:
            raise ValueError(f"Model not supported: {model_name}")


    def create_corpus(self):
        print(f"Creating corpus: {self.corpus_name}")
        for corpus in rag.list_corpora():
            if corpus.display_name == self.corpus_name:
                self.rag_corpus = corpus
                print(f"✅ Found existing corpus: {corpus.name}")
                return
        self.rag_corpus = rag.create_corpus(
            display_name=self.corpus_name,
            description=f"A test corpus where we test RAG on the {self.corpus_name} dataset",
            embedding_model_config=self.embedding_model_config,
        )
        print(f"Corpus created: {self.rag_corpus}")
        

    def count_files_in_gcs_bucket(self, gcs_path: str) -> int:
        """
        Counts the number of files in a Google Cloud Storage path,
        excluding directories and hidden files.

        Args:
        gcs_path: The full GCS path, including the bucket name and any prefix.
        * Example: 'gs://my-bucket/my-folder'

        Returns:
        The number of files in the GCS path.
        """


        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)

        count = 0
        blobs = bucket.list_blobs(prefix=self.bucket_prefix)
        for blob in blobs:
            if not blob.name.endswith("/") and not any(
                part.startswith(".") for part in blob.name.split("/")
            ):  # Exclude directories and hidden files
                count += 1

        return count


    def count_directories_after_split(self, gcs_path: str) -> int:
        """
        Counts the number of directories in a Google Cloud Storage path.

        Args:
        gcs_path: The full GCS path, including the bucket name and any prefix.

        Returns:
        The number of directories in the GCS path.
        """
        num_files_in_path = self.count_files_in_gcs_bucket(gcs_path)
        num_directories = math.ceil(num_files_in_path / 10000)
        return num_directories


    def import_rag_files_from_gcs(self, paths: list[str], chunk_size: int, chunk_overlap: int) -> None:
        """Imports files from Google Cloud Storage to a RAG corpus.

        Args:
            paths: A list of GCS paths to import files from.
            chunk_size: The size of each chunk to import.
            chunk_overlap: The overlap between consecutive chunks.
            corpus_name: The name of the RAG corpus to import files into.

        Returns:
            None
        """
        total_imported, total_num_of_files = 0, 0

        for path in paths:
            num_files_to_be_imported = self.count_files_in_gcs_bucket(path)
            total_num_of_files += num_files_to_be_imported
            max_retries, attempt, imported = 10, 0, 0
            while attempt < max_retries and imported < num_files_to_be_imported:
                response = rag.import_files(
                    self.rag_corpus.name,
                    [path],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    timeout=20000,
                    max_embedding_requests_per_min=1400,
                )
                imported += response.imported_rag_files_count or 0
                attempt += 1
            total_imported += imported

        print(f"{total_imported} files out of {total_num_of_files} imported!")
        
    
    def convert_beir_to_rag_corpus(self, corpus: dict[str, dict[str, str]], output_dir: str) -> None:
        """
        Convert a BEIR corpus to Vertex RAG corpus format with a maximum of 10,000
        files per subdirectory.

        For each document in the BEIR corpus, we will create a new txt where:
        * doc_id will be the file name
        * doc_content will be the document text prepended by title (if any).

        Args:
        corpus: BEIR corpus
        output_dir (str): Directory where the converted corpus will be saved

        Returns:
        None (will write output to disk)
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        file_count, subdir_count = 0, 0
        current_subdir = os.path.join(output_dir, f"{subdir_count}")
        os.makedirs(current_subdir, exist_ok=True)

        # Convert each file in the corpus
        for doc_id, doc_content in corpus.items():
            # Combine title and text (if title exists)
            full_text = doc_content.get("title", "")
            if full_text:
                full_text += "\n\n"
            full_text += doc_content["text"]

            # Create a new file for each file.
            file_path = os.path.join(current_subdir, f"{doc_id}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(full_text)

            file_count += 1

            # Create a new subdirectory if the current one has reached the limit
            if file_count >= 10000:
                subdir_count += 1
                current_subdir = os.path.join(output_dir, f"{subdir_count}")
                os.makedirs(current_subdir, exist_ok=True)
                file_count = 0

        print(f"Conversion complete. {len(corpus)} files saved in {output_dir}")
    
        
    def index_corpus(self, dataset_name: str, corpus: List[Dict[str, Any]], chunk_size: int = 1024, chunk_overlap: int = 128):
        local_output_dir = f"./data/vertex_ai_corpus/{self.corpus_name}"
        if not corpus:
            print("⚠️ Empty corpus provided. Skipping indexing.")
            return

        print(f"📥 Converting and indexing {len(corpus)} documents to Vertex AI RAG corpus: {self.corpus_name}")

        # Step 1: Convert to BEIR-style dict format
        beir_format = {doc["id"]: {"title": doc.get("title", ""), "text": doc["content"]} for doc in corpus}

        # Step 2: Convert to .txt files in subdirectories
        # self.convert_beir_to_rag_corpus(beir_format, output_dir=local_output_dir)

        # Step 3: Upload subdirectories to GCS with progress bar
        print(f"☁️ Uploading converted files to GCS bucket: {self.gcs_bucket_name}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)

        gcs_paths = []
        for subdir in sorted(os.listdir(local_output_dir)):
            subdir_path = os.path.join(local_output_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            gcs_prefix = f"{self.bucket_prefix}/{subdir}/"
            files = os.listdir(subdir_path)
            with tqdm(total=len(files), desc=f"Uploading {subdir}", unit="file") as pbar:
                for filename in files:
                    local_path = os.path.join(subdir_path, filename)
                    blob_path = gcs_prefix + filename
                    # blob = bucket.blob(blob_path)
                    # blob.upload_from_filename(local_path)
                    pbar.update(1)

            gcs_paths.append(f"gs://{self.gcs_bucket_name}/{self.bucket_prefix}/{subdir}/")
            print(f"✅ Uploaded directory {subdir} to GCS")

        # Step 4: Import uploaded files into Vertex RAG Engine
        print(f"📤 Importing {len(gcs_paths)} directories to Vertex RAG corpus: {self.corpus_name}")
        self.import_rag_files_from_gcs(paths=gcs_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    def retrieve(self, query: str, top_k: int = 10):
        # Retrieval not yet implemented
        pass


