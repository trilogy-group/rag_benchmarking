from enum import Enum

class PineconeNativeEmbeddingModel(Enum):
    LLAMA_V2 = "llama-text-embed-v2"
    MULTILINGUAL_E5_LARGE = "multilingual-e5-large"

class OpenAIEmbeddingModel(Enum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

class CohereEmbeddingModel(Enum):
    COHERE_EMBEDDING_V4 = "embed-v4.0"
    COHERE_EMBEDDING_V3 = "embed-multilingual-v3.0"

class GeminiEmbeddingModel(Enum):
    GEMINI_001 = "gemini-embedding-001"
    GEMINI_EXP_03_07 = "gemini-embedding-exp-03-07"
    
class StellaEmbeddingModel(Enum):
    STELLA_1_5B_V5 = "dunzhang/stella_en_1.5B_v5"

class ModernBERTEmbeddingModel(Enum):
    MODERN_BERT_LARGE = "answerdotai/ModernBERT-large"