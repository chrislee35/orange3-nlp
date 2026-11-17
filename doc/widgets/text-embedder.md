# Text Embedder

Performs embedding on text.

**Inputs**

- Corpus: documents to embed
- Embedder: an embedding model widget

**Outputs**

- Embedded Corpus: the input corpus with `emb_0 … emb_N` feature columns

Applies the connected embedder to every document, in batches of 32, and appends the resulting vector as numeric
`emb_*` columns.

Connect an embedding model widget ([Doc2Vec](doc2vec-embedder.md), [E5](e5-embedder.md),
[FastText](fasttext-embedder.md), [Gemini](gemini-embedder.md), [Nomic](nomic-embedder.md),
[OpenAI](openai-embedder.md), [SBERT](sbert-embedder.md), [spaCy](spacy-embedder.md) or [USE](use-embedder.md)).
If a corpus arrives with no embedder connected, the widget shows the error *You must connect an embedder*.

The result can be used with Orange's clustering and modeling widgets, or sent to the
[Reference Library](reference-library.md).

![Text embedding models](../../imgs/nlp-embedder-lineup.png)
