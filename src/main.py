from dotenv import load_dotenv
from experiments import get_experiment_config
from experiment_runner import ExperimentRunner


load_dotenv()




def main():
    experiment = get_experiment_config("frame-3-large-gpt-4o")
    index_name = experiment.index_name
    agent_name = experiment.agent_name
    datastore = experiment.datastore(index_name=index_name, agent_name=agent_name, text_embedding_model=experiment.text_embedding_model, openai_model=experiment.openai_model)
    dataset = experiment.dataset(dataset_name=experiment.dataset_name)
    retriever = experiment.retriever(index_name=index_name, agent_name=agent_name)
    evaluator = experiment.evaluator(outfile_name=f"{experiment.name}.csv")
    

    print(f"Dataset: {dataset}")
    experiment_runner = ExperimentRunner(datastore=datastore, dataset=dataset, retriever=retriever, evaluator=evaluator)
    experiment_runner.run()


if __name__ == "__main__":
    main()