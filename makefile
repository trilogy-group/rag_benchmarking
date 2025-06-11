.PHONY: download download_index evaluate full shell clean

# Default experiment name (can be overridden)
EXP ?= QDRANT_FRAME_GEMINI_001_GPT_4O

# Run interactive shell
shell:
	@echo "=== [Shell] Starting interactive experiment shell ==="
	poetry run python src/shell.py

# Download only the dataset
download:
	@echo "=== [Download] Dataset for experiment: $(EXP) ==="
	poetry run python src/main.py --task download --experiment $(EXP)

# Download and index the dataset
download_index:
	@echo "=== [Download + Index] Dataset for experiment: $(EXP) ==="
	poetry run python src/main.py --task download_index --experiment $(EXP)

# Run evaluation only
evaluate:
	@echo "=== [Evaluate] Running evaluation for experiment: $(EXP) ==="
	poetry run python src/main.py --task evaluate --experiment $(EXP)

# Full experiment: download, index, and evaluate
full:
	@echo "=== [Full] Running complete pipeline for experiment: $(EXP) ==="
	poetry run python src/main.py --task full --experiment $(EXP)

# Clean up temporary files
clean:
	@echo "=== [Clean] Removing temporary files ==="
	@rm -rf __pycache__ *.pyc *.csv

metrics:
	@echo "=== [Metrics] Calculating metrics for experiment: $(EXP) ==="
	poetry run python src/metrics.py 



