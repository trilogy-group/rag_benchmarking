import argparse
from dotenv import load_dotenv
from experiments import get_experiment_config, ExperimentName
from experiment_runner import ExperimentRunner

load_dotenv()


def download_dataset(experiment):
    print(f"Downloading dataset: {experiment.dataset_name}")
    dataset = experiment.dataset(dataset_name=experiment.dataset_name)
    dataset.load()
    print(f"Downloaded {len(dataset.corpus)} documents and {len(dataset.queries)} queries.")


def download_and_index_dataset(experiment):
    print(f"Downloading and indexing dataset: {experiment.dataset_name}")
    dataset = experiment.dataset(dataset_name=experiment.dataset_name)
    dataset.load()
    corpus_items = list(dataset.corpus.items())[:experiment.max_corpus_size]

    documents = [
        {
            "id": doc_id,
            "content": f"{doc.get('title', '')}. {doc.get('text', '')}"
        }
        for doc_id, doc in corpus_items
    ]

    datastore = experiment.datastore(
        index_name=f"{experiment.name.value}-index",
        agent_name=f"{experiment.name.value}-agent",
        text_embedding_model=experiment.text_embedding_model,
        openai_model=experiment.openai_model,
        namespace=experiment.name.value
    )

    print(f"Indexing {len(documents)} documents into datastore...")
    datastore.index_corpus(documents)


def evaluate(experiment):
    print(f"Evaluating retriever on: {experiment.name}")
    dataset = experiment.dataset(dataset_name=experiment.dataset_name)
    dataset.load()

    retriever = experiment.retriever(
        index_name=f"{experiment.name.value}-index",
        agent_name=f"{experiment.name.value}-agent",
        namespace=experiment.name.value,
        text_embedding_model=experiment.text_embedding_model
    )

    evaluator = experiment.evaluator(outfile_name=f"{experiment.name.value}.csv")

    experiment_runner = ExperimentRunner(
        dataset=dataset,
        datastore=None,  # No reindexing needed here
        retriever=retriever,
        evaluator=evaluator,
        max_corpus_size=experiment.max_corpus_size
    )

    evaluator.evaluate(retriever, dataset, experiment.max_corpus_size)


def run_experiment():
    experiment = get_experiment_config(ExperimentName.PINECONE_BEIR_NQ_GEMINI_001_GPT_4O)

    print(f"Running full experiment: {experiment.name}\n")
    download_and_index_dataset(experiment)
    evaluate(experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG benchmarking experiments.")
    parser.add_argument("--task", type=str, choices=["download", "download_index", "evaluate", "full"], required=True)
    parser.add_argument("--experiment", type=str, required=True)

    args = parser.parse_args()
    experiment = get_experiment_config(ExperimentName[args.experiment])

    if args.task == "download":
        download_dataset(experiment)
    elif args.task == "download_index":
        download_and_index_dataset(experiment)
    elif args.task == "evaluate":
        evaluate(experiment)
    elif args.task == "full":
        download_and_index_dataset(experiment)
        evaluate(experiment)
