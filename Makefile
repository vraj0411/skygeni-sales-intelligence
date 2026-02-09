.PHONY: install train eda dashboard test clean all

# Default target
all: install train dashboard

# Install dependencies
install:
	pip install -r requirements.txt

# Run EDA
eda:
	python notebooks/01_eda.py

# Train model
train:
	python train.py

# Run dashboard
dashboard:
	streamlit run app.py

# Run tests
test:
	pytest tests/ -v --cov=src

# Clean artifacts
clean:
	rm -rf models/
	rm -rf data/processed/
	rm -rf reports/
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Full pipeline
pipeline: eda train

# Help
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make eda        - Run exploratory data analysis"
	@echo "  make train      - Train the risk scoring model"
	@echo "  make dashboard  - Launch Streamlit dashboard"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Remove generated files"
	@echo "  make pipeline   - Run full pipeline (eda + train)"
