from typing import Any

from AnyQt.QtWidgets import QComboBox, QLabel
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget
from Orange.widgets.widget import Input, Output
from Orange.data import Domain, StringVariable, Table
from orangecontrib.text.corpus import Corpus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter
import numpy as np

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class TextChunkerWorker(QThread):
    result = pyqtSignal(Corpus)
    progress = pyqtSignal(int)

    def __init__(self, corpus: Corpus, chunk_size: int, chunk_strategy: str) -> None:
        super().__init__()
        self.corpus = corpus
        self.chunk_size = chunk_size
        self.chunk_strategy = chunk_strategy
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        corpus = self.corpus
        chunk_size = self.chunk_size
        splitter: Any
        if self.chunk_strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.1))
        elif self.chunk_strategy == "semantic":
            splitter = TextSplitter(chunk_size)
        else:
            raise ValueError(f"Unknown chunk strategy: {self.chunk_strategy}")
        documents = []
        metas = []
        n_docs = len(corpus.documents)
        for idx, doc in enumerate(corpus.documents):
            chunks: list[str]
            if hasattr(splitter, "split_text"):
                chunks = splitter.split_text(doc)
            elif hasattr(splitter, "chunks"):
                chunks = splitter.chunks(doc)
            else:
                raise TypeError(f"Unsupported splitter: {type(splitter).__name__}")
            documents.extend(chunks)
            metas.extend([[idx]] * len(chunks))
            self.progress.emit(int((idx + 1) / n_docs * 100))
        excerpt_var = StringVariable("excerpt")
        original_idx_var = StringVariable("source_doc_idx")
        domain = Domain([], metas=[excerpt_var, original_idx_var])
        metas_array = np.array([[text, str(meta[0])] for text, meta in zip(documents, metas)], dtype=object)
        table = Table.from_numpy(domain, X=np.empty((len(metas_array), 0)), metas=metas_array)
        chunked_corpus = Corpus.from_table(domain, table)
        chunked_corpus.attributes = corpus.attributes.copy()
        chunked_corpus.set_text_features([chunked_corpus.domain["excerpt"]])
        self.result.emit(chunked_corpus)


class OWTextChunker(widget.OWWidget):
    name = _tr.m[73, "Text Chunker"]
    description = _tr.m[67, "Splits text documents into smaller chunks."]
    icon = "icons/nlp-chunk.svg"
    priority = 100

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Chunked Corpus", Corpus)

    chunk_size = setting(256)
    chunk_strategy = setting("recursive")
    want_main_area = False

    def __init__(self) -> None:
        super().__init__()
        self.corpus = None
        self.worker = None
        self.progress_bar_value = 0
        self.layout_control_area()

    def layout_control_area(self) -> None:
        layout = layout_of(self.controlArea)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        layout.addWidget(QLabel(_tr.m[11, "Chunking Strategy"]))
        self.chunk_strategy_combo = QComboBox()
        self.chunk_strategy_combo.addItems(["recursive", "semantic"])
        self.chunk_strategy_combo.setCurrentText(self.chunk_strategy)
        self.chunk_strategy_combo.currentTextChanged.connect(self.on_chunk_strategy_change)
        layout.addWidget(self.chunk_strategy_combo)
        layout.addWidget(QLabel(_tr.m[10, "Chunk size:"]))
        self.chunk_size_combo = QComboBox()
        self.chunk_size_combo.addItems(["128", "256", "512", "1024"])
        self.chunk_size_combo.setCurrentText(str(self.chunk_size))
        self.chunk_size_combo.currentTextChanged.connect(self.on_chunk_size_change)
        layout.addWidget(self.chunk_size_combo)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    @Inputs.data
    def set_data(self, data: Corpus) -> None:
        self.corpus = data
        self.apply()

    def on_chunk_size_change(self, val: int) -> None:
        self.chunk_size = int(val)
        self.apply()

    def on_chunk_strategy_change(self, val: str) -> None:
        self.chunk_strategy = val
        self.apply()

    def apply(self) -> None:
        if not self.corpus:
            self.Outputs.data.send(None)
            return
        self.progressBarInit()
        self.worker = TextChunkerWorker(self.corpus, self.chunk_size, self.chunk_strategy)
        self.worker.progress.connect(self.progressBarSet)
        self.worker.result.connect(self.finished_chunking)
        self.setStatusMessage("Chunking...")
        self.setBlocking(True)
        self.worker.start()

    def finished_chunking(self, result: Corpus) -> None:
        self.progressBarFinished()
        self.setStatusMessage("")
        self.setBlocking(False)
        if result:
            self.Outputs.data.send(result)
        else:
            self.Outputs.data.send(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    corpus = Corpus("book-excerpts")
    WidgetPreview(OWTextChunker).run(corpus)
