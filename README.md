# orange3-nlp

This provides a collection of widgets for Natural Language Processing.

## Installation

Within the Add-ons installer, click on "Add more..." and type in orange3-nlp

## Widgets

![Canvas with 8 major widgets provided by the Orange3-NLP package](imgs/nlp-widget-lineup.png)

* General Widgets
  * Abstractive Summary
  * Extractive Summary
  * Named Entity Recognition
  * POS Tagger
  * POS Viewer
  * Question Answering
  * Reference Library
  * Ollama RAG

![Text Splitting Widgets](imgs/nlp-text-splitting.png)

* Text Splitting Widgets
  * Text Chunker
  * Tokens to Corpus

![Text Embedding Models](imgs/nlp-embedder-lineup.png)

* Text Embedding Models
  * Doc2Vec
  * E5
  * FastText
  * Gemini
  * Nomic
  * OpenAI
  * Sentence Embedder (SBERT)
  * spaCy
  * USE

![Training widget for Doc2Vec embedder](imgs/nlp-train-doc2vec.png)

* Training of Text Embedding Widget
  * Train Doc2Vec

![Polish sentiment analysis widget, Analiza Sentymentu](imgs/nlp-analiza-sentymentu.png)

* For Polish Sentiment Analysis
  * Analiza Sentymentu

### Summary Widgets

- **Extractive Summary**: Selects and joins key sentences or phrases from the original text.

![Extractive Summary of The Little Match-Seller](imgs/extractive-summary.png)

- **Abstractive Summary**: Generates new sentences that paraphrase and condense the original content (more similar to how humans summarize).

![Abstractive Summary of The Litle Match-Seller](imgs/abstractive-summary.png)

### Named Entity Recognition

**Named Entity Recognition (NER)** is a task in NLP that locates and classifies named entities in text into predefined categories such as:

- **PERSON** – names of people  
- **ORG** – organizations  
- **GPE** – countries, cities, or locations  
- **DATE**, **TIME**, **MONEY**, etc.

### Part of Speech Tagging

Part-of-speech (POS) tagging assigns grammatical categories to each word in a sentence.

#### Common POS Tags

| Tag | Meaning       | Example        |
|-----|---------------|----------------|
| NN  | Noun          | `cat`, `city`  |
| VB  | Verb          | `run`, `is`    |
| JJ  | Adjective     | `fast`, `red`  |
| RB  | Adverb        | `quickly`      |
| DT  | Determiner    | `the`, `an`    |
| IN  | Preposition   | `on`, `with`   |

> POS tagging is essential for syntactic parsing and downstream NLP tasks.

#### Part of Speech Viewer

This uses spaCy's displacy HTML renderer to provide a parsed dependency tree of the parts of speech of the input text.

![Part of Speech Viewer with parsed Slovenian text.](imgs/pos-viewer.png)

### Question Answering

**Question Answering (QA)** systems aim to extract or generate answers to user questions from a text or knowledge base.

![Question and Answers for "Who Died?" against the Book Excerpts corpus](imgs/qa.png)

### Text Splitting Widgets

#### Tokens to Corpus

The Tokens to Corpus widget takes the tokens from the Preprocess Text widgets.

![Tokens to Corpus workflow](imgs/nlp-tokens-to-corpus-workflow.png)

#### Text Chunker

Text Chunker supports 2 chunking strategies to split text.  The first is [LangChain's RecursiveCharacterTextSplitter](https://lagnchain.readthedocs.io/en/stable/modules/indexes/text_splitters/examples/recursive_text_splitter.html) and the second is [semantic-text-splitter](https://pypi.org/project/semantic-text-splitter/).

![Text Chunker widget](imgs/nlp-text-chunker.png)


### Reference Augmented Generation

**Reference Augmented Generation (RAG)** is a method of enhancing large language model (LLM) responses by *providing external documents as supporting context*. Instead of relying solely on the model's training data, RAG:

- **Retrieves** relevant snippets from a document collection (knowledge base).
- **Augments** the prompt to the LLM by including this retrieved content.
- **Generates** a more accurate and grounded answer based on the context.

![RAG Workflow](imgs/nlp-rag-workflow.png)

Let's take a look at the Reference Library

![Reference Library](imgs/nlp-reference-library.png)

And lastly, let's look at the Ollama RAG use.

![Ollama RAG Widget: Using the phi Ollama model, and a prompt of "Who were the Munchins and what are they good at?"](imgs/nlp-ollama-rag.png)

### Polish Sentiment Analysis

Since Polish sentiment analysis support in Orange was limited, Analiza Sentymentu provides a tuned model.

![Polish sentiment analysis workflow](imgs/nlp-analiza-sentymenty-workflow.png)
