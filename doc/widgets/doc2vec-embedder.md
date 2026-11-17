# Doc2Vec Embedder

Provides the Gensim Doc2Vec model

**Inputs**

- none

**Outputs**

- Embedder: an embedding model for Text Embedder or Reference Library

Provides a [gensim](https://radimrehurek.com/gensim/) Doc2Vec model. Pick a model from the **Model** list; the list holds
the `.bin` models in Orange's data directory (`gensim` folder), including those saved by
[Train Doc2Vec](train-doc2vec.md). Press **Rescan Models** to refresh the list after adding a model. Problems loading a
model are shown under *Model Errors*.
