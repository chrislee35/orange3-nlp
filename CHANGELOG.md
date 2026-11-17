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
- **Ollama-based Entity-Relationship Extraction** widget: extracts subject-verb-object relations with a local Ollama model using structured JSON output. Produces the same output table as the rule-based widget. Includes an editable prompt and Extract/Cancel buttons.
- New `nlp-relations.svg` icon for the entity-relationship extraction widgets.
- Internationalization support: all widgets now use Orange's `Translator` for their names, descriptions and UI strings.
  - New `orangecontrib/nlp/i18n/` with `English.json` and `Japanese.json`.
  - The translation files are now included in the built package (`package-data`).

### Changed
- The following widgets were updated for translation, with some incidental formatting cleanup (quote style, blank lines, import layout):
  - Models and embedders: Doc2Vec, E5, FastText, Gemini, Nomic, OpenAI, SBERT, spaCy, Universal Sentence Encoder, Text Embedder, Train Doc2Vec.
  - Summarization and Q&A: Abstractive Summary, Extractive Summary, Question Answer, Ollama RAG.
  - Text processing and analysis: Analiza Sentymentu, NER, POS Tagger, POS Viewer, Reference Library, Text Chunker, Token Corpus.
- Both entity-relationship extraction widgets are translated (new entries 87-97 appended to the i18n files, with Japanese translations).
- Code quality: all ruff errors fixed (including the Translator preamble moved below the imports), type hints added throughout (ruff `ANN` rules now enforced), Pylance/pyright clean, and the code formatted with black (line length 140). Added `[tool.black]`, `[tool.ruff]` and `[tool.pyright]` configuration and a `typings/` stub package that points `AnyQt` at PyQt5 for type checkers (excluded from the wheel).
- New helpers: `util/qt_helpers.py` (`layout_of`, `box_layout_of`), `util/typed_settings.py` (`setting`, a typed wrapper for Orange `Setting`), and `util/paths.py` (`orange_data_dir`).
- Removed `orangecontrib/__init__.py` (a deprecated `pkg_resources` namespace declaration); `orangecontrib` is now a PEP 420 namespace package.
- The gemini and OpenAI embedder widgets' `update()` method (which overrode `QWidget.update`) is renamed `update_embedder()`.
- Version bumped from 0.0.8 to 0.0.9.

### Fixed
- The Ollama option in the Named Entity Recognition widget ignored the configured host, port and model (it always used `localhost:11434` and `phi`); it now uses the widget settings and has a request timeout.
- The Rule-based Entity-Relationship Extraction widget referenced a nonexistent icon (`RelationExtraction.svg`).
- `os.path.makedirs` (nonexistent) in the UDPipe downloader; now `os.makedirs`.
- The OpenAI Embedder widget constructed `OpenAIEmbedder(api_key, model)`, which raised a `TypeError`; the embedder reads its own API key.
- Text Embedder: Cancel/stop raised `AttributeError` because the worker had no `cancel()`.
- Text Chunker: an unknown chunking strategy raised an unbound-variable error; it now raises a clear `ValueError`.
