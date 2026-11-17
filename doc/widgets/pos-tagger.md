# POS Tagger

Tag parts of speech using different frameworks.

**Inputs**

- Corpus: documents to tag

**Outputs**

- Tagged Corpus: the input corpus with a *POS Tags* column

Part-of-speech (POS) tagging assigns a grammatical category to each word. Choose the framework with the radio buttons:

- **spaCy**
- **Stanza**
- **UDPipe**

The model for the corpus language is downloaded the first time it is needed. The *POS Tags* column holds a JSON list
with, for every token, its `text`, `lemma`, `pos`, `tag`, `dep` (dependency relation) and `head` (offset to the head
token). Connect the output to [POS Viewer](pos-viewer.md) to see the result as a dependency tree.

| Tag | Meaning | Example |
|-----|---------|---------|
| NN | Noun | `cat`, `city` |
| VB | Verb | `run`, `is` |
| JJ | Adjective | `fast`, `red` |
| RB | Adverb | `quickly` |
| DT | Determiner | `the`, `an` |
| IN | Preposition | `on`, `with` |
