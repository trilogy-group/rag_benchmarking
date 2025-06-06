from dotenv import load_dotenv
from experiments import get_experiment_config, ExperimentName
from experiment_runner import ExperimentRunner


load_dotenv()




def main():
    # experiment = get_experiment_config("beir-covid-3-large-gpt-4o")
    # experiment = get_experiment_config("beir-scidocs-3-large-gpt-4o")
    # experiment = get_experiment_config("frame-3-large-gpt-4o")
    # experiment = get_experiment_config("ragtruth-3-large-gpt-4o")
    # experiment = get_experiment_config(ExperimentName.PINECONE_FRAME_LLAMAV2_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_BEIR_COVID_LLAMAV2_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_BEIR_SCIDOCS_LLAMAV2_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_FRAME_E5LARGE_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_BEIR_SCIDOCS_E5LARGE_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_FRAME_3_LARGE_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_FRAME_COHERE_V4_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_FRAME_GEMINI_001_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_FRAME_GEMINI_EXP_03_07_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.PINECONE_FRAME_STELLA_1_5B_V5_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.QDRANT_BEIR_COVID_3_LARGE_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.QDRANT_BEIR_SCIDOCS_3_LARGE_GPT_4O)
    experiment = get_experiment_config(ExperimentName.QDRANT_BEIR_SCIDOCS_COHERE_V4_GPT_4O)

    # experiment = get_experiment_config(ExperimentName.QDRANT_FRAME_3_LARGE_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.QDRANT_FRAME_COHERE_V4_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.QDRANT_FRAME_GEMINI_001_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.QDRANT_FRAME_GEMINI_EXP_03_07_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.QDRANT_FRAME_STELLA_1_5B_V5_GPT_4O)
    # experiment = get_experiment_config(ExperimentName.QDRANT_FRAME_MODERN_BERT_LARGE_GPT_4O)

    

    
    print(f"Running Experiment: {experiment.name}\n\n")

    index_name = f"{experiment.name.value}-index" 
    agent_name = f"{experiment.name.value}-agent"
    namespace = experiment.name.value

    datastore = experiment.datastore(
        index_name=index_name, 
        agent_name=agent_name, 
        text_embedding_model=experiment.text_embedding_model, 
        openai_model=experiment.openai_model,
        namespace=namespace
    )
    dataset = experiment.dataset(dataset_name=experiment.dataset_name)
    retriever = experiment.retriever(index_name=index_name, agent_name=agent_name, namespace=namespace, text_embedding_model=experiment.text_embedding_model)
    evaluator = experiment.evaluator(outfile_name=f"{experiment.name.value}.csv")
    
    print(f"Dataset: {dataset}")
    experiment_runner = ExperimentRunner(datastore=datastore, dataset=dataset, retriever=retriever, evaluator=evaluator)
    experiment_runner.run()


if __name__ == "__main__":
    main()