from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox,
    QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QWidget
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import Domain, StringVariable, Table
from orangecontrib.text.corpus import Corpus
import numpy as np
import uuid
import faiss
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import requests
from orangecontrib.nlp.util.spacy_downloader import SpaCyDownloader

from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB(QThread):
    result = pyqtSignal(object)  # emits the built VectorDB
    progress = pyqtSignal(int)     # emits progress (0-100)

    def __init__(self, corpus:Corpus):
        super().__init__()
        self.corpus = corpus
        self.index = None
        self._cancelled = False

    def run(self):
        # Identify the embedding columns by name
        embedding_attrs = [attr for attr in self.corpus.domain.attributes if attr.name.startswith("emb_")]

        # Sort by index (in case they're out of order)
        embedding_attrs.sort(key=lambda a: int(a.name.split("_")[1]))

        # Get the column indices for the embeddings
        embedding_indices = [self.corpus.domain.index(attr) for attr in embedding_attrs]

        # Extract the relevant columns (corpus.X is a 2D numpy array or similar)
        embeddings = self.corpus.X[:, embedding_indices]

        vectors = np.vstack(embeddings)
        #faiss.normalize_L2(vectors) # need to figure out why this is throwing an error
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)

        self._cancelled = False
        batch_size = 256
        num_vectors = vectors.shape[0]
        total_batches = (num_vectors + batch_size - 1) // batch_size
        idx = 0
        last_progress = 0
        for i in range(0, num_vectors, batch_size):
            self.index.add(vectors[i:i + batch_size])
            idx += 1
            progress = int(100*(idx/total_batches))
            if progress > last_progress:
                self.progress.emit(progress)
                last_progress = progress
            if self._cancelled:
                self.result.emit(None)
                return

        self.result.emit(self.index)

    def search(self, query_vec, top_k=5):
        if self.index is None:
            return []
        #faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, top_k)
        return [(self.corpus.documents[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.corpus.documents)]

    def cancel(self):
        self._cancelled = True

class SearchWorker(QThread):
    result = pyqtSignal(list)
    progress = pyqtSignal(int)

    def __init__(self, query_vec, vector_db, top_k):
        super().__init__()
        self.query_vec = query_vec
        self.vector_db = vector_db
        self.top_k = top_k
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        self.progress.emit(10)
        results = self.vector_db.search(self.query_vec, top_k=self.top_k)
        self.progress.emit(100)
        if not self._cancelled:
            self.result.emit(results)

class OWReferenceLibrary(widget.OWWidget):
    name = "Reference Library"
    description = "Stores documents in a vector database and retrieves references."
    icon = "icons/nlp-reference.svg"
    priority = 150

    class Inputs:
        data = Input("Corpus", Corpus)
        embedder = Input("Embedder", object, auto_summary=False)

    class Outputs:
        data = Output("Excerpts", Corpus)

    max_excerpts = settings.Setting(5)
    threshold = settings.Setting(0.0)
    query = settings.Setting("")

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.vector_db = None
        self.embed_func = None
        self.worker = None

        self.layout_control_area()
        self.layout_main_area()

    def layout_control_area(self):
        max_excerpts_label = QLabel("Max excerpts:")
        self.controlArea.layout().addWidget(max_excerpts_label)
        self.max_excerpts_spin = QSpinBox()
        self.max_excerpts_spin.setRange(1, 100)
        self.max_excerpts_spin.setValue(self.max_excerpts)
        self.max_excerpts_spin.valueChanged.connect(self.on_max_excerpts_change)
        self.controlArea.layout().addWidget(self.max_excerpts_spin)

        threshold_label = QLabel("Matching threshold:")
        self.controlArea.layout().addWidget(threshold_label)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(self.on_threshold_change)
        self.controlArea.layout().addWidget(self.threshold_spin)
        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def layout_main_area(self):
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter query here")
        self.query_input.setText(self.query)
        self.query_input.returnPressed.connect(self.on_query_change)
        self.mainArea.layout().addWidget(self.query_input)

        buttons_layout = QHBoxLayout()
        self.search_button = QPushButton("Find References")
        self.search_button.clicked.connect(self.on_query_change)
        buttons_layout.addWidget(self.search_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_worker)
        buttons_layout.addWidget(self.stop_button)
        self.mainArea.layout().addLayout(buttons_layout)

        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.mainArea.layout().addWidget(self.results_display)

    def on_max_excerpts_change(self, val):
        self.max_excerpts = val

    def on_threshold_change(self, val):
        self.threshold = val

    def on_query_change(self):
        self.query = self.query_input.text()
        self.find_references()

    @Inputs.data
    def set_data(self, data):
        self.corpus = data
        self.build_vector_db()

    @Inputs.embedder
    def set_embedder(self, embedder):
        self.embed_func = embedder
        self.find_references()

    def build_vector_db(self):
        if not self.corpus:
            return
        self.vector_db = VectorDB(self.corpus)
        self.worker = self.vector_db
        self.progressBarInit()
        self.worker.progress.connect(self.progressBarSet)
        self.worker.result.connect(self.finish_vector_db_indexing)
        self.worker.start()

    def finish_vector_db_indexing(self, index: object):
        self.progressBarFinished()
        self.find_references()

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            self.progressBarInit()

    def find_references(self):
        if not self.corpus or not self.query or not self.embed_func:
            return

        self.stop_worker()

        self.progressBarInit()
        query_vec = self.embed_func([self.query])
        self.worker = SearchWorker(query_vec, self.vector_db, top_k=self.max_excerpts)
        self.worker.progress.connect(self.progressBarSet)
        self.worker.result.connect(self.display_results)
        self.worker.start()
        
    def display_results(self, results):
        excerpt_var = StringVariable("excerpt")
        score_var = StringVariable("score")
        metas = [excerpt_var, score_var]

        new_rows = []
        for text, score in results:
            if score < self.threshold:
                continue
            new_metas = [text, f"{score:.4f}"]
            new_rows.append(new_metas)

        if not new_rows:
            self.results_display.setPlainText("No results above the threshold.")
            return

        domain = Domain([], metas=metas)
        metas_array = np.array(new_rows, dtype=object)

        table = Table.from_numpy(domain, X=np.empty((len(new_rows), 0)), metas=metas_array)

        new_corpus: Corpus = Corpus.from_table(domain, table)
        new_corpus.attributes['language'] = self.corpus.attributes['language']
        new_corpus.set_text_features([new_corpus.columns.excerpt])

        self.results_display.setPlainText("\n---\n".join([f"{r[0]}\n(Similarity: {r[1]:.4f})" for r in results if r[1] >= self.threshold]))
        self.Outputs.data.send(new_corpus)
        self.progressBarFinished()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.nlp.widgets.owtext_embedder import EmbedderFactory

    corpus = Corpus('tests/book-excerpts-embedded.tab')
    embed_func = EmbedderFactory.get_embedder("sentence-transformers")

    w = WidgetPreview(OWReferenceLibrary)
    w.create_widget()
    w.send_signals(set_data=corpus, set_embedder=embed_func)
    w.run()
