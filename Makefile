# Makefile for building Sphinx docs and Python package

.PHONY: all docs build clean

# Build everything: docs + package
all: docs build

# Build Sphinx documentation
docs:
	sphinx-build -b html doc/ doc/_build/html

# Build the Python package (wheel and source)
build:
	python -m build

# Push package to pypi.org
release: build
	twine upload dist/*

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info doc/_build

test:
	python tests/test_nlp.py TestWidgets.test_abstractive_summary
	python tests/test_nlp.py TestWidgets.test_extractive_summary
	python tests/test_nlp.py TestWidgets.test_owner
	python tests/test_nlp.py TestWidgets.test_postagger
	python tests/test_nlp.py TestWidgets.test_question_answer
	python tests/test_nlp.py TestWidgets.test_reference_library
	python tests/test_nlp.py TestWidgets.test_ollama_rag