from typing import Any

from AnyQt.QtWidgets import QLabel
from AnyQt.QtCore import QThread, pyqtSignal
from Orange.widgets import widget
from Orange.widgets.widget import Input, Output
from Orange.data import Domain, StringVariable, Table
from orangecontrib.text.corpus import Corpus
import numpy as np

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class TokenCorpusWorker(QThread):
    result = pyqtSignal(Corpus)
    progress = pyqtSignal(int)

    def __init__(self, corpus: Corpus) -> None:
        super().__init__()
        self.corpus = corpus
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        corpus = self.corpus
        title = None
        for col in corpus.domain.metas:
            if "title" in col.name.lower():
                title = col
        documents = []
        titles = []
        n_docs = len(corpus.tokens)
        for idx, doc in enumerate(corpus.tokens):
            for token in doc:
                if title:
                    row: Any = corpus[idx]
                    titles.append(row[title.name].value)
                documents.append(token)
            self.progress.emit(int((idx + 1) / n_docs * 100))
        token_var = StringVariable("Token")
        if title:
            domain = Domain([], metas=[token_var, title])
            metas_array = np.array(list(zip(documents, titles)), dtype=object)
        else:
            domain = Domain([], metas=[token_var])
            metas_array = np.array(list(zip(documents)), dtype=object)
        table = Table.from_numpy(domain, X=np.empty((len(metas_array), 0)), metas=metas_array)
        token_corpus = Corpus.from_table(domain, table)
        token_corpus.attributes = corpus.attributes.copy()
        token_corpus.set_text_features([token_corpus.domain["Token"]])
        self.result.emit(token_corpus)


class OWTokenCorpus(widget.OWWidget):
    name = _tr.m[76, "Tokens to Corpus"]
    description = _tr.m[72, "Takes tokens from Preprocess Text and outputs them as a Corpus."]
    icon = "icons/nlp-token2corpus.svg"
    priority = 100

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Tokenized Corpus", Corpus)

    chunk_size = setting(256)
    want_main_area = False

    def __init__(self) -> None:
        super().__init__()
        self.corpus = None
        self.worker = None
        layout_of(self.controlArea).addWidget(QLabel(_tr.m[75, "Tip: If you want to split on paragraphs, use regexp: [^\\n]+"]))

    @Inputs.data
    def set_data(self, data: Corpus) -> None:
        self.corpus = data
        self.apply()

    def apply(self) -> None:
        if not self.corpus:
            self.Outputs.data.send(None)
            return
        self.progressBarInit()
        self.worker = TokenCorpusWorker(self.corpus)
        self.worker.progress.connect(self.progressBarSet)
        self.worker.result.connect(self.finished)
        self.setStatusMessage("Converting Tokens to Corpus...")
        self.setBlocking(True)
        self.worker.start()

    def finished(self, result: Corpus) -> None:
        self.progressBarFinished()
        self.setStatusMessage("")
        self.setBlocking(False)
        if result:
            self.Outputs.data.send(result)
        else:
            self.Outputs.data.send(None)
