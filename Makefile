# SARA - Makefile
# Streamlined commands for setup, data preparation, and running the application

.PHONY: run help setup rag extract clean explorer data-check data-extract dependencies

# Run the SARA application (default target)
run:
	@echo "Starting SARA application..."
	streamlit run Home.py

# Help target
help:
	@echo "SARA - Summarize. Analyze. Retrieve. Annotate."
	@echo ""
	@echo "Available commands:"
	@echo "  make setup        - Set up dependencies and environment"
	@echo "  make run          - Run the SARA application"
	@echo "  make explorer     - Run the data explorer dashboard only"
	@echo "  make rag          - Build the RAG vector database"
	@echo "  make data-check   - Check if data is properly installed"
	@echo "  make data-extract - Extract the data archive if present"
	@echo "  make clean        - Clean temporary files"
	@echo "  make help         - Show this help message"

# Setup dependencies with uv (recommended)
setup:
	@echo "Setting up SARA environment..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for dependency installation..."; \
		uv sync; \
	else \
		echo "uv not found, using pip instead..."; \
		pip install -r requirements.txt; \
	fi
	@echo "Setup complete! Run 'make data-check' to verify data installation."

# Alternative: Setup with pip
dependencies:
	@echo "Installing dependencies with pip..."
	pip install -r requirements.txt

# No longer needed - definition moved to be the default target

# Run the explorer dashboard only
explorer:
	@echo "Starting data explorer dashboard..."
	streamlit run pages/01_Explorer.py

# Build the RAG vector database
rag:
	@echo "Building RAG vector database..."
	@if [ ! -d "data/articles_clean" ]; then \
		echo "Error: Data not found. Run 'make data-check' first."; \
		exit 1; \
	fi
	@mkdir -p rag
	python rag/create_rag.py
	@echo "RAG database built successfully!"

# Check if data is properly installed
data-check:
	@echo "Checking for required data..."
	@if [ ! -d "data" ]; then \
		echo "Error: Data directory not found!"; \
		echo "Please download and extract the data archive."; \
		exit 1; \
	fi
	@if [ ! -f "data/metadata.csv" ]; then \
		echo "Error: metadata.csv not found!"; \
		exit 1; \
	fi
	@if [ ! -d "data/articles_clean" ]; then \
		echo "Error: articles_clean directory not found!"; \
		exit 1; \
	fi
	@echo "Data check passed! Data is properly installed."
	@echo "Found $(shell ls -l data/articles_clean | wc -l) article files."

# Extract data from zip archive if present
data-extract:
	@echo "Looking for data archive..."
	@if [ -f "data.zip" ]; then \
		echo "Extracting data.zip..."; \
		python -m zipfile -e data.zip .; \
		echo "Data extraction complete!"; \
	else \
		echo "Error: data.zip not found!"; \
		echo "Please download the data archive first."; \
		exit 1; \
	fi

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf .ipynb_checkpoints
	rm -rf */.ipynb_checkpoints
	@echo "Clean complete!"