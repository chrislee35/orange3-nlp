# Named Entity Recognition

Uses selected NER framework to extract named entities.

**Inputs**

- Corpus: documents to analyze

**Outputs**

- Corpus with Entities: the input corpus with a *Named Entities* column

Named Entity Recognition (NER) locates and classifies named entities in text into categories such as **PERSON**,
**ORG** (organizations), **GPE** (countries, cities and other locations), **DATE**, **TIME** and **MONEY**.

The widget reads the text left by the transformation stages of a preceding *Preprocess Text* widget (lowercase, remove accents, parse HTML, remove URLs); the tokenization, normalization and filtering stages have no effect. Entity recognition relies on capital letters, so if you use *Preprocess Text* before this widget, untick **Lowercase**. Without *Preprocess Text* the raw document text is used.

Choose a framework with the radio buttons:

- **NLTK**
- **spaCy** (`en_core_web_sm`)
- **Flair**
- **Ollama**. The model is asked, with Ollama's `format: json` option and temperature 0, to return a JSON list of entities labeled PERSON, ORG, GPE, DATE, MONEY or OTHER. Small models label less reliably than spaCy, NLTK or Flair, and are much slower; a model of about 8 billion parameters (for example `llama3.1:8b`) is a sensible minimum. Documents the reply cannot be parsed for are reported as `Error: ...`.

Each entity is written to the *Named Entities* column as `text (LABEL)`, separated by commas.

**Ollama settings.** Selecting *Ollama* shows the server **Host** (default `localhost`), **Port** (default `11434`) and
**Model**. The model list is read from the server; change the host or port and the list refreshes. The model must already
be pulled on the server (`ollama pull <model>`).
