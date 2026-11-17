# Nomic Embedder

Provides the Ollama nomic embedding model

**Inputs**

- none

**Outputs**

- Embedder: an embedding model for Text Embedder or Reference Library

Provides embeddings from the `nomic-embed-text` model served by [Ollama](https://ollama.com). Set the **Host** and
**Port** of the server (default `localhost:11434`) and make sure the model is available
(`ollama pull nomic-embed-text`). Vectors are L2-normalized.
