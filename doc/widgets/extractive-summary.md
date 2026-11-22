# Extractive Summary

Extract summaries from text using various frameworks.

**Inputs**

- Corpus: documents to summarize

**Outputs**

- Corpus with Summaries: the input corpus with an *Extractive Summary* column

Extractive summaries select and join the most important sentences from the original text.

Choose a framework with the radio buttons:

- **Sumy**
- **Summa** (TextRank)
- **BART**
- **Ollama**, using a local large language model.

Only **Sumy** and **Summa** are extractive: every sentence of their summary comes from the document. **BART** and **Ollama** write new text, as in *Abstractive Summary*, so their summaries are not made of the original sentences. With Ollama the model is only *asked* to extract, and it usually paraphrases.

Press **Cancel** to stop a running job.

By default the widget summarizes as soon as a corpus arrives and whenever you choose another framework. Summarizing is slow,
and the neural models are reloaded for every run, so untick **Apply automatically** while you are still changing the
input or the framework. The widget then waits until you press **Apply**. Ticking the box again runs any pending change.
Sending new input stops a run that is still going.

**Ollama settings.** Selecting *Ollama* shows the server **Host** (default `localhost`), **Port** (default `11434`) and
**Model**. The model list is read from the server; change the host or port and the list refreshes. The model must already
be pulled on the server (`ollama pull <model>`).

![Extractive Summary of The Little Match-Seller](../../imgs/extractive-summary.png)
