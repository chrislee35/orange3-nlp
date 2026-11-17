# Changelog

## 0.0.9 (unreleased)

### Dependencies
- Require Orange3 >= 3.41.0 (was >= 3.39.0).
- Require numpy >= 2.0 (was >= 1.26.4) so the project installs cleanly alongside Orange3 3.41.0 on Windows and Linux.
- Require gensim >= 4.4.0 (was >= 4.3.3); 4.3.x caps numpy below 2.0.
- Replace `fasttext` with `fasttext-predict >= 0.9.2.4`. `fasttext` 0.9.3 has no wheels (it must be compiled on Windows) and is not numpy 2 compatible. `fasttext-predict` ships Windows and Linux wheels and imports as `fasttext`. It supports inference only, which is all the FastText Embedder widget uses (`load_model`, `get_word_vector`, `get_dimension`).
- Fix a missing closing quote and comma on the `Orange3-Text >= 1.16.3` dependency line in `pyproject.toml`.

### Added
- **Rule-based Entity-Relationship Extraction** widget: extracts subject-verb-object relations using spaCy.
- Internationalization support: all widgets now use Orange's `Translator` for their names, descriptions and UI strings.
  - New `orangecontrib/nlp/i18n/` with `English.json` and `Japanese.json`.
  - The translation files are now included in the built package (`package-data`).

### Changed
- The following widgets were updated for translation, with some incidental formatting cleanup (quote style, blank lines, import layout):
  - Models and embedders: Doc2Vec, E5, FastText, Gemini, Nomic, OpenAI, SBERT, spaCy, Universal Sentence Encoder, Text Embedder, Train Doc2Vec.
  - Summarization and Q&A: Abstractive Summary, Extractive Summary, Question Answer, Ollama RAG.
  - Text processing and analysis: Analiza Sentymentu, NER, POS Tagger, POS Viewer, Reference Library, Text Chunker, Token Corpus.
- Version bumped from 0.0.8 to 0.0.9.
