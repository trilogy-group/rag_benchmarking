from enum import Enum

class PineconeNativeEmbeddingModel(Enum):
    LLAMA_V2 = "llama-text-embed-v2"
    MULTILINGUAL_E5_LARGE = "multilingual-e5-large"

class OpenAIEmbeddingModel(Enum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

