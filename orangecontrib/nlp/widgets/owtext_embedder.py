from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox,
    QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QWidget
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import Domain, StringVariable, Table, ContinuousVariable
from orangecontrib.text.corpus import Corpus
import numpy as np

class EmbedderWorker(QThread):
    result = pyqtSignal(Corpus)  # emits the embedded Corpus
    progress = pyqtSignal(int)   # emits progress (0-100)

    def __init__(self, corpus: Corpus, embed_func):
        super().__init__()
        self.corpus = corpus
        self.embed_func = embed_func

    def run(self):
        last_progress = 0
        batch_size = 32  # You can adjust this based on memory/performance
        total_documents = len(self.corpus.documents)
        total_batches = (total_documents + batch_size - 1) // batch_size
        vectors = []
        idx = 0
        for i in range(0, total_documents, batch_size):
            batch = self.corpus.documents[i:i + batch_size]
            vecs = self.embed_func(batch)
            vectors.extend(vecs)
            idx += 1
            progress = int(100*(idx/total_batches))
            if progress > last_progress:
                self.progress.emit(progress)
                last_progress = progress

        embeddings = np.array(vectors)
        dim = embeddings.shape[1]
        embedding_attrs = [ContinuousVariable(f"emb_{i}") for i in range(dim)]

        # Combine original X with embeddings
        new_X = np.hstack((self.corpus.X, embeddings))

        # Combine domains
        original_attrs = self.corpus.domain.attributes
        combined_attrs = list(original_attrs) + embedding_attrs

        domain = Domain(
            attributes=combined_attrs,
            class_vars=self.corpus.domain.class_vars,
            metas=self.corpus.domain.metas
        )

        new_table = Table(domain, new_X, self.corpus.Y, self.corpus.metas)
        new_corpus = Corpus.from_table(domain, new_table)

        self.result.emit(new_corpus)    

class EmbedderWrapper(object):
    def __init__(self, embedder):
        self.embedder = embedder
    
    def embed(self, texts: list[str]):
        return self.embedder(texts)

class OWTextEmbedder(widget.OWWidget):
    name = "Text Embedder"
    description = "Performs embedding on text."
    icon = "icons/nlp-embed.svg"
    priority = 150

    class Inputs:
        data = Input("Corpus", Corpus)
        embedder = Input("Embedder", object, auto_summary=False)

    class Outputs:
        data = Output("Embedded Corpus", Corpus)

    want_main_area = False
    want_control_area = False

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.worker = None
        #self.layout_control_area()

    def layout_control_area(self):
        self.controlArea.layout().addWidget(QLabel("Embedder:"))
        self.embedder_combo = QComboBox()
        self.embedder_combo.addItems([
            "sentence-transformers", "e5-small-v2", "nomic-embed-text", "spacy"
        ])
        self.embedder_combo.setCurrentText(self.embedder)
        self.embedder_combo.currentTextChanged.connect(self.on_embedder_change)
        self.controlArea.layout().addWidget(self.embedder_combo)

        # Ollama host/port config panel (initially hidden)
        self.ollama_panel = QWidget()
        ollama_layout = QVBoxLayout()
        self.ollama_panel.setLayout(ollama_layout)

        self.host_input = QLineEdit(self.ollama_host)
        self.port_input = QLineEdit(self.ollama_port)
        self.host_input.setPlaceholderText("Ollama Host")
        self.port_input.setPlaceholderText("Ollama Port")

        ollama_layout.addWidget(QLabel("Ollama Host:"))
        ollama_layout.addWidget(self.host_input)
        ollama_layout.addWidget(QLabel("Ollama Port:"))
        ollama_layout.addWidget(self.port_input)

        self.controlArea.layout().addWidget(self.ollama_panel)
        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def on_embedder_change(self, text):
        self.embedder = text
        self.run_embedder()

    @Inputs.data
    def set_data(self, data):
        self.corpus = data
        self.run_embedder()

    def run_embedder(self):
        if not (self.corpus and self.embedder):
            return
        embed_func = EmbedderFactory.get_embedder(self.embedder)
        self.Outputs.embedder.send(embed_func)
        self.worker = EmbedderWorker(self.corpus, embed_func)
        self.progressBarInit()
        self.worker.progress.connect(self.progressBarSet)
        self.worker.result.connect(self.finish_embedding)
        self.worker.start()

    def finish_embedding(self, results: Corpus):
        self.progressBarFinished()
        self.Outputs.data.send(results)

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            self.progressBarInit()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    corpus = Corpus('book-excerpts')
    WidgetPreview(OWTextEmbedder).run(corpus)
