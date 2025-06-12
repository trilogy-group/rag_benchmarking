from retrievers._retriever import Retriever
from typing import List, Tuple
import os
from vertexai import rag
from vertexai.rag import retrieval_query, RagResource

class VertexAIRetriever(Retriever):
    def __init__(self, index_name: str, agent_name: str, namespace: str, text_embedding_model: str):
        self.corpus_name = index_name  # RAG corpus display name or ID
        self.agent_name = agent_name
        self.namespace = namespace
        self.text_embedding_model = text_embedding_model
        self.rag_corpus = self.get_corpus_by_display_name(self.corpus_name)

    def get_corpus_by_display_name(self, display_name: str):
        for corpus in rag.list_corpora():
            if corpus.display_name == display_name:
                return corpus
        raise ValueError(f"RAG corpus '{display_name}' not found.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        print(f"🔍 Vertex RAG API Retrieval | Query: '{query}' | Top K: {top_k}")
        
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=top_k,
            filter=rag.Filter(vector_distance_threshold=0.5),
            # ranking=rag.Ranking(
            #     llm_ranker=rag.LlmRanker(
            #         model_name="gemini-2.0-flash"
            #     )
            # )
        )

        response = retrieval_query(
            rag_resources=[RagResource(rag_corpus=self.rag_corpus.name)],
            text=query,
            rag_retrieval_config=rag_retrieval_config
        )

        print(f"🔍 Response: {response}")   

        hits = {}

        for context in response.contexts.contexts:
            source_uri = context.source_uri or "unknown"
            score = context.score

            # Extract doc_id from GCS URI: the filename without `.txt`
            # Example: "gs://bucket/path/abc123.txt" → "abc123"
            doc_id = os.path.splitext(os.path.basename(source_uri))[0]
            hits[doc_id] = score

        print(f"Hits: {hits}")

        print(f"✅ Retrieved {len(hits)} contexts from Vertex RAG corpus")
        return hits