# spaCy Embedder

Provides the spaCy embedding model

**Inputs**

- none

**Outputs**

- Embedder: an embedding model for Text Embedder or Reference Library

Embeds each document as the vector of its spaCy `Doc` (the average of its token vectors). The spaCy model for the corpus
language is downloaded automatically. The widget has no settings.
