from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox,
    QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QWidget
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import Domain, StringVariable, Table, ContinuousVariable
from orangecontrib.text.corpus import Corpus
from orangecontrib.nlp.util.embedder_models import EmbedderModel
import numpy as np

class EmbedderWorker(QThread):
    result = pyqtSignal(Corpus)  # emits the embedded Corpus
    progress = pyqtSignal(int)   # emits progress (0-100)

    def __init__(self, corpus: Corpus, embedder: EmbedderModel):
        super().__init__()
        self.corpus = corpus
        self.embedder = embedder

    def run(self):
        last_progress = 0
        batch_size = 32  # You can adjust this based on memory/performance
        total_documents = len(self.corpus.documents)
        total_batches = (total_documents + batch_size - 1) // batch_size
        vectors = []
        idx = 0
        for i in range(0, total_documents, batch_size):
            batch = self.corpus.documents[i:i + batch_size]
            vecs = self.embedder.embed(self.corpus.language, batch)
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
            attributes = combined_attrs,
            class_vars = self.corpus.domain.class_vars,
            metas      = self.corpus.domain.metas
        )

        new_table = Table(domain, new_X, self.corpus.Y, self.corpus.metas)
        new_corpus = Corpus.from_table(domain, new_table)

        self.result.emit(new_corpus)

class OWTextEmbedder(widget.OWWidget):
    name = "Text Embedder"
    description = "Performs embedding on text."
    icon = "icons/nlp-embed.svg"
    priority = 150

    class Inputs:
        data = Input("Corpus", Corpus)
        embedder = Input("Embedder", EmbedderModel, auto_summary=False)

    class Outputs:
        data = Output("Embedded Corpus", Corpus)

    want_main_area = False
    want_control_area = False

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.worker = None
        self.embedder = None

    @Inputs.data
    def set_data(self, data):
        self.corpus = data
        self.run_embedder()

    @Inputs.embedder
    def set_embedder(self,embedder):
        self.embedder = embedder
        self.run_embedder()
    
    def run_embedder(self):
        if self.corpus and not self.embedder:
            self.Error.add_message("You much connect an embedder")
            return
        if not (self.corpus and self.embedder):
            return
        self.worker = EmbedderWorker(self.corpus, self.embedder)
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
