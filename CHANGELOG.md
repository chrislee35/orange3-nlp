# Changelog

## 0.0.10 (unreleased)

### Added
- **Ollama-based Entity-Relationship Extraction** widget: extracts subject-verb-object relations with a local Ollama model using structured JSON output. Produces the same output table as the rule-based widget. Includes an editable prompt and Extract/Cancel buttons.
- New `nlp-relations.svg` icon for the entity-relationship extraction widgets.
- A Sphinx documentation site in `doc/` (one help page per widget, `widgets.json`, Makefile and `make.bat`), written in Markdown with MyST. The `doc` extra now installs `myst-parser` instead of `recommonmark`. Build with `cd doc && make html`; the built HTML is bundled into the wheel and used by the widgets' Help button.
- Both entity-relationship extraction widgets are translated (new entries 87-97 appended to the i18n files, with Japanese translations).
- `.flake8` configuration (140 columns, E203 ignored for black compatibility).

### Changed
- Restrict `requires-python` to `>=3.11,<3.14`: TensorFlow publishes no Python 3.14 wheels, so installing on 3.14 failed with an unresolvable dependency error. Dependencies resolve on Linux and Windows for Python 3.11, 3.12 and 3.13, and every compiled dependency has a Windows wheel (the only source-only packages, such as `summa`, `baycomp` and `python-louvain`, are pure Python).
- Code quality: all ruff errors fixed (including the Translator preamble moved below the imports), type hints added throughout (ruff `ANN` rules now enforced), Pylance/pyright clean, and the code formatted with black (line length 140). Added `[tool.black]`, `[tool.ruff]` and `[tool.pyright]` configuration and a `typings/` stub package that points `AnyQt` at PyQt5 for type checkers (excluded from the wheel).
- New helpers: `util/qt_helpers.py` (`layout_of`, `box_layout_of`), `util/typed_settings.py` (`setting`, a typed wrapper for Orange `Setting`), and `util/paths.py` (`orange_data_dir`).
- Removed `orangecontrib/__init__.py` (a deprecated `pkg_resources` namespace declaration); `orangecontrib` is now a PEP 420 namespace package.
- The Gemini and OpenAI embedder widgets' `update()` method (which overrode `QWidget.update`) is renamed `update_embedder()`.
- README rewritten: installation (pip, requirements, models and services, API keys), all widgets including the entity-relationship and Text Embedder widgets, localization, and a development section; fixed typos and a broken LangChain link, and "Reference Augmented Generation" is now "Retrieval-Augmented Generation".
- Packaging: the built Sphinx HTML is now bundled via `setup.py` (the previous `data-files` glob in `pyproject.toml` could not match directories), and `doc`, `build`, `dist` and `typings` are excluded from package discovery.

### Fixed
- FastText Embedder failed with `'_FastText' object has no attribute 'get_word_vector'` (`fasttext-predict` is prediction-only). The widget now loads the Facebook `.bin` models with gensim (`load_facebook_vectors`), which is already a dependency, supports out-of-vocabulary words, and needs no separate fasttext package.
- The Ollama option in the Named Entity Recognition widget ignored the configured host, port and model (it always used `localhost:11434` and `phi`); it now uses the widget settings and has a request timeout.
- The Rule-based Entity-Relationship Extraction widget referenced a nonexistent icon (`RelationExtraction.svg`).
- `os.path.makedirs` (nonexistent) in the UDPipe downloader; now `os.makedirs`.
- The OpenAI Embedder widget constructed `OpenAIEmbedder(api_key, model)`, which raised a `TypeError`; the embedder reads its own API key.
- Text Embedder: Cancel/stop raised `AttributeError` because the worker had no `cancel()`.
- Text Chunker: an unknown chunking strategy raised an unbound-variable error; it now raises a clear `ValueError`.

## 0.0.9

### Dependencies
- Require Orange3 >= 3.41.0 (was >= 3.39.0).
- Require numpy >= 2.0 (was >= 1.26.4) so the project installs cleanly alongside Orange3 3.41.0 on Windows and Linux.
- Require gensim >= 4.4.0 (was >= 4.3.3); 4.3.x caps numpy below 2.0.
- Remove the `fasttext` dependency (0.9.3 has no wheels and is not numpy 2 compatible). 0.0.9 briefly used `fasttext-predict`, which turned out to lack word vectors; see 0.0.10.
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
- Translation progress files (`*.progress.json`) are now git-ignored.
- Version bumped from 0.0.8 to 0.0.9.
