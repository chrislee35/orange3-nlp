# Ollama-based Entity-Relationship Extraction

Extract subject–verb–object relations using an Ollama model

**Inputs**

- Corpus: documents to analyze

**Outputs**

- Relations: a table of subject–verb–object triples

Asks a local [Ollama](https://ollama.com) model to extract subject–verb–object triples from each document. The
request uses Ollama's structured JSON output with temperature 0, so results are repeatable for a given model.

**Controls**

- **Host** and **Port** of the Ollama server (default `localhost:11434`).
- **Model**: the models available on the server.
- **Prompt**: the instruction sent with each document. The default asks for short noun-phrase subjects and objects, a
  short lower-case verb, and the supporting sentence. You can edit it; it is saved with the workflow.
- **Extract** and **Cancel**. The widget does not run when data arrives, because LLM extraction is slow on large
  corpora; press **Extract**.

Documents that fail are listed in the *Errors* box and the remaining documents are still processed.

The widget outputs a table with one row per relation and these columns:

| Column | Meaning |
|--------|---------|
| `input_chunk_id` | Index of the document the relation came from |
| `subject` | The subject entity |
| `verb` | The relation: a verb, or a verb plus preposition such as `migrate_to` |
| `object` | The object entity |
| `sentence` | The sentence supporting the relation |

Small models sometimes leave pronouns as subjects (for example `She`). A more specific prompt or a larger model
improves this.
