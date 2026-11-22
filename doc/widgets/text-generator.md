# Text Generator

Generate text from a prompt with a local language model.

**Inputs**

- Corpus (optional): documents to use in the prompt template

**Outputs**

- Corpus: one row per generation, with the input columns plus `Prompt`, `Generated Text` and `Sample`

The widget sends a prompt to a language model and returns what the model writes. What happens next depends on the kind of model. A **base model** (such as `gpt2`) simply *continues* the text, so `The capital of France is` is followed by more text about France. An **instruction-tuned model** (such as `phi4-mini` in Ollama) treats the prompt as a request and answers it. Use the widget to see how prompts, temperature and chance change what a model writes.

**Backends**

- **Ollama**: a model running on a local Ollama server. Set the **Host** (default `localhost`), **Port** (default `11434`) and **Model**. The model list is read from the server and the box stays editable if the server is not reachable. The model must already be pulled (`ollama pull <model>`). The request is sent to Ollama's `/api/generate` endpoint, which applies the model's own prompt template, so instruction-tuned models answer instead of continuing. The text streams into the output pane.
- **Hugging Face**: a small causal language model that runs in Python through `transformers`. The default is `gpt2`, which is downloaded the first time it is used (about 0.5 GB) and kept in memory afterwards. Type any other causal-language-model name, such as `distilgpt2`. `gpt2` and `distilgpt2` are base models: they do not follow instructions, they only continue text.

**Prompt template**

Without an input, the template is the prompt and it runs once. With a corpus connected, it runs once per document, and these placeholders are replaced:

- `{text}`: the document's text (the corpus's text features joined together)
- `{ColumnName}`: the value of any other column, for example `{category}` or `{Title}`

A `{name}` that matches no column is left as written, and the widget shows a warning, so braces in code or JSON in a prompt are safe. Press **Generate** to run (generation is slow, so it never starts by itself) and **Stop** to cancel.

**Parameters**

- **Temperature**: how random the choice of each next word is. `0` always picks the most likely word, which is repeatable but tends to repeat itself. Higher values give more varied text.
- **Max new tokens**: the length limit for each generation.
- **Top-p**: sample only from the most likely words that together cover this fraction of the probability (ignored at temperature `0`).
- **Fixed seed**: makes a run repeatable. Sample *i* of a prompt uses the seed plus *i*, so several samples differ from each other but the whole run can be repeated exactly. Without a fixed seed, every run is different.
- **Samples per prompt**: how many times to generate from each prompt. Use more than one to see how much the output varies.

**Output**

Each output row holds the input row's columns (if any), the rendered `Prompt`, the `Generated Text` and the `Sample` number (starting at 1). `Generated Text` is the output corpus's text feature, so it can go straight into *Text Embedder*, *Preprocess Text* or *Corpus Viewer*. A generation that fails is reported as `Error: ...` in its row and does not stop the others.

Because every output has the same `Prompt`, `Generated Text` and `Sample` columns, you cannot connect one *Text Generator* to the input of another: Orange rejects the duplicate column names. To run several prompts on the same documents, connect the *Corpus* to one *Text Generator* for each prompt.
