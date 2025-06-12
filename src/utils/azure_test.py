from azure.identity import DefaultAzureCredential, get_bearer_token_provider, EnvironmentCredential
import os
from azure.search.documents.indexes.models import SearchIndex, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration, AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters, SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchIndexingBufferedSender
import requests
from azure.search.documents.indexes.models import KnowledgeAgent, KnowledgeAgentAzureOpenAIModel, KnowledgeAgentTargetIndex, KnowledgeAgentRequestLimits, AzureOpenAIVectorizerParameters
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import KnowledgeAgentRetrievalRequest, KnowledgeAgentMessage, KnowledgeAgentMessageTextContent, KnowledgeAgentIndexParams
import textwrap
import json


load_dotenv()


endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
credential = DefaultAzureCredential()
# credential = DefaultAzureCredential()
# credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
# credential = EnvironmentCredential()

token_provider = get_bearer_token_provider(credential, "https://search.azure.com/.default")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_gpt_deployment = "gpt-4o"
azure_openai_gpt_model = "gpt-4o"

# azure_openai_api_version = "2025-03-01-preview"
azure_openai_embedding_deployment = "text-embedding-3-large"
azure_openai_embedding_model = "text-embedding-3-large"
index_name = "earth_at_foo"
agent_name = "earth-search-agent"
answer_model = "gpt-4o"
# api_version = "2025-05-01-Preview"




index = SearchIndex(
    name=index_name,
    fields=[
        SearchField(name="id", type="Edm.String", key=True, filterable=True, sortable=True, facetable=True),
        SearchField(name="page_chunk", type="Edm.String", filterable=False, sortable=False, facetable=False),
        SearchField(name="page_embedding_text_3_large", type="Collection(Edm.Single)", stored=False, vector_search_dimensions=3072, vector_search_profile_name="hnsw_text_3_large"),
        SearchField(name="page_number", type="Edm.Int32", filterable=True, sortable=True, facetable=True)
    ],
    vector_search=VectorSearch(
        profiles=[VectorSearchProfile(name="hnsw_text_3_large", algorithm_configuration_name="alg", vectorizer_name="azure_openai_text_3_large")],
        algorithms=[HnswAlgorithmConfiguration(name="alg")],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="azure_openai_text_3_large",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=azure_openai_endpoint,
                    deployment_name=azure_openai_embedding_deployment,
                    model_name=azure_openai_embedding_model
                )
            )
        ]
    ),
    semantic_search=SemanticSearch(
        default_configuration_name="semantic_config",
        configurations=[
            SemanticConfiguration(
                name="semantic_config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[
                        SemanticField(field_name="page_chunk")
                    ]
                )
            )
        ]
    )
)

index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
index_client.create_or_update_index(index)
print(f"Index '{index_name}' created or updated successfully")





url = "https://raw.githubusercontent.com/Azure-Samples/azure-search-sample-data/refs/heads/main/nasa-e-book/earth-at-night-json/documents.json"
documents = requests.get(url).json()

with SearchIndexingBufferedSender(endpoint=endpoint, index_name=index_name, credential=credential) as client:
    client.upload_documents(documents=documents)

print(f"Documents uploaded to index '{index_name}'")




agent = KnowledgeAgent(
    name=agent_name,
    models=[
        KnowledgeAgentAzureOpenAIModel(
            azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                resource_url=azure_openai_endpoint,
                deployment_name=azure_openai_gpt_deployment,
                model_name=azure_openai_gpt_model,
                api_key=os.getenv("AZURE_OPENAI_API_KEY") 
            )
        )
    ],
    target_indexes=[
        KnowledgeAgentTargetIndex(
            index_name=index_name,
            default_reranker_threshold=2.5
        )
    ],
)

index_client.create_or_update_agent(agent)
print(f"Knowledge agent '{agent_name}' created or updated successfully")


instructions = """
An Q&A agent that can answer questions about the Earth at night.
Sources have a JSON format with a ref_id that must be cited in the answer.
If you do not have the answer, respond with "I don't know".
"""

# messages = [
#     {
#         "role": "assistant",
#         "content": instructions
#     }
# ]



agent_client = KnowledgeAgentRetrievalClient(endpoint=endpoint, agent_name=agent_name, credential=credential)

# messages.append({
#     "role": "user",
#     "content": """
#     Why do suburban belts display larger December brightening than urban cores even though absolute light levels are higher downtown?
#     Why is the Phoenix nighttime street grid is so sharply visible from space, whereas large stretches of the interstate between midwestern cities remain comparatively dim?
#     """
# })

instructions = """
    A Q&A agent that can answer questions about the Earth at night.
    Sources have a JSON format with a ref_id that must be cited in the answer.
    If you do not have the answer, respond with "I don't know".
    """

messages = [
    KnowledgeAgentMessage(
        role="assistant",
        content=[KnowledgeAgentMessageTextContent(text=instructions)]
    ),
    KnowledgeAgentMessage(
        role="user",
        content=[KnowledgeAgentMessageTextContent(text="""
        Why do suburban belts display larger December brightening than urban cores even though absolute light levels are higher downtown?
        Why is the Phoenix nighttime street grid is so sharply visible from space, whereas large stretches of the interstate between midwestern cities remain comparatively dim?                                          
        """)]
    )
]

try:
    
    print("Retrieving...")
    # Define index params with reranker threshold (optional: tweak as needed)
    index_params = [
        KnowledgeAgentIndexParams(
            index_name=index_name,
            reranker_threshold=2.5  # or whatever makes sense for your eval
        )
    ]

    try:
        retrieval_result = agent_client.knowledge_retrieval.retrieve(
            retrieval_request=KnowledgeAgentRetrievalRequest(
                messages=messages,
                target_index_params=index_params
            )
        )       
    except Exception as e:
        print(f"Error retrieving agentic response: {e}")


    # retrieval_result = agent_client.knowledge_retrieval.retrieve(
    #     retrieval_request=KnowledgeAgentRetrievalRequest(
    #         messages=[KnowledgeAgentMessage(role=msg["role"], content=[KnowledgeAgentMessageTextContent(text=msg["content"])]) for msg in messages],
    #         target_index_params=[KnowledgeAgentIndexParams(index_name=index_name, reranker_threshold=2.5)]
    #     )
    # )
    print("Retrieval result ")
    print(retrieval_result)
except Exception as e:
    print(e)



print("Response")
print(textwrap.fill(retrieval_result.response[0].content[0].text, width=120))

print("Activity")
print(json.dumps([a.as_dict() for a in retrieval_result.activity], indent=2))

print("Results")
print(json.dumps([r.as_dict() for r in retrieval_result.references], indent=2))