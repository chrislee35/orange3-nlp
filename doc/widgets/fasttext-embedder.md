# FastText Embedder

Provides the FastText embedding model

**Inputs**

- none

**Outputs**

- Embedder: an embedding model for Text Embedder or Reference Library

Provides [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) Common Crawl vectors (300 dimensions) for the corpus
language. A document is embedded as the average of its word vectors. The model for a language is downloaded to Orange's
data directory the first time it is used; **these files are very large (several GB)**. The widget has no settings.
