from AnyQt.QtWidgets import QComboBox, QLabel, QSizePolicy, QVBoxLayout
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import Domain, StringVariable, Table
from orangecontrib.text.corpus import Corpus
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

class TextChunkerWorker(QThread):
    result = pyqtSignal(Corpus)
    progress = pyqtSignal(int)

    def __init__(self, corpus: Corpus, chunk_size: int):
        super().__init__()
        self.corpus = corpus
        self.chunk_size = chunk_size
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        corpus = self.corpus
        chunk_size = self.chunk_size

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1)
        )

        documents = []
        metas = []

        n_docs = len(corpus.documents)
        for idx, doc in enumerate(corpus.documents):
            chunks = splitter.split_text(doc)
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
    name = "Text Chunker"
    description = "Splits text documents into smaller chunks."
    icon = "icons/nlp-chunk.svg"
    priority = 100

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Chunked Corpus", Corpus)

    chunk_size = settings.Setting(256)
    want_main_area = False

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.worker = None
        self.progress_bar_value = 0
        self.layout_control_area()
    
    def layout_control_area(self):
        # UI setup
        layout = self.controlArea.layout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        layout.addWidget(QLabel("Chunk size:"))
        self.chunk_size_combo = QComboBox()
        self.chunk_size_combo.addItems(["128", "256", "512", "1024"])
        self.chunk_size_combo.setCurrentText(str(self.chunk_size))
        self.chunk_size_combo.currentTextChanged.connect(self.on_chunk_size_change)
        layout.addWidget(self.chunk_size_combo)
        layout.setAlignment(Qt.AlignTop)

    @Inputs.data
    def set_data(self, data: Corpus):
        self.corpus = data
        self.apply()

    def on_chunk_size_change(self, val):
        self.chunk_size = int(val)
        self.apply()

    def apply(self):
        if not self.corpus:
            self.Outputs.data.send(None)
            return

        self.progressBarInit()
        self.worker = TextChunkerWorker(self.corpus, self.chunk_size)
        self.worker.progress.connect(self.progressBarSet)
        self.worker.result.connect(self.finished_chunking)
        self.setStatusMessage("Chunking...")
        self.setBlocking(True)
        self.worker.start()


    def finished_chunking(self, result: Corpus):
        self.progressBarFinished()
        self.setStatusMessage("")
        self.setBlocking(False)
        if result:
            self.Outputs.data.send(result)
        else:
            self.Outputs.data.send(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    corpus = Corpus('book-excerpts')
    WidgetPreview(OWTextChunker).run(corpus)
