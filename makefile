# Makefile for running experiment functions via Poetry with src layout

PYTHON=poetry run python
SCRIPT=main
PYTHONPATH=src

.PHONY: run_experiment
run_experiment:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c "from $(SCRIPT) import run_experiment; run_experiment()"

.PHONY: download_dataset
download_dataset:
ifndef DATASET
	$(error DATASET is not set. Use 'make download_dataset DATASET=your_dataset_name')
endif
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c "from $(SCRIPT) import download_dataset; download_dataset('$(DATASET)')"

.PHONY: download_and_index_dataset
download_and_index_dataset:
ifndef DATASET
	$(error DATASET is not set. Use 'make download_and_index_dataset DATASET=your_dataset_name')
endif
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c "from $(SCRIPT) import download_and_index_dataset; download_and_index_dataset('$(DATASET)')"