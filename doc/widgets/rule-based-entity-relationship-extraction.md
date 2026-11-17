# Rule-based Entity-Relationship Extraction

Extract subject–verb–object relations using spaCy

**Inputs**

- Corpus: documents to analyze

**Outputs**

- Relations: a table of subject–verb–object triples

Extracts relations from the dependency parse produced by spaCy (English model). It runs automatically whenever a
corpus arrives.

The rules handle:

- active and passive verbs (`temperature influences growth`, `growth is influenced by temperature`),
- prepositional relations such as `migrate_to` and `linked_to`,
- copular relations, reported with the verb `is`.

Discourse verbs such as *show*, *suggest*, *indicate*, *find* and *report* are ignored because they rarely describe a
relation between entities.

The widget outputs a table with one row per relation and these columns:

| Column | Meaning |
|--------|---------|
| `input_chunk_id` | Index of the document the relation came from |
| `subject` | The subject entity |
| `verb` | The relation: a verb, or a verb plus preposition such as `migrate_to` |
| `object` | The object entity |
| `sentence` | The sentence supporting the relation |

The output has the same columns as the [Ollama-based Entity-Relationship Extraction](ollama-entity-relationship-extraction.md)
widget, so either can be used in the same workflow.
