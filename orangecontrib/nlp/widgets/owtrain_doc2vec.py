from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from orangecontrib.text.corpus import Corpus
from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.misc.environ import data_dir_base
import numpy as np
import multiprocessing
import os
import datetime

class Doc2VecEmbedder(EmbedderModel):
    def __init__(self, model: Doc2Vec):
        self.model = model

    def embed_doc2vec(self, model, text: str) -> np.ndarray:
        tokens = simple_preprocess(text)
        vector = model.infer_vector(tokens)
        return np.array(vector, dtype=np.float32)

    def embed(self, language, texts: list[str]) -> np.ndarray:
         return np.vstack([self.embed_doc2vec(self.model, text) for text in texts])

    def save(self, *args, **kwargs):
        print(*args)
        self.model.save(*args, **kwargs)

class Doc2VecTrainingWorker(QThread):
    result = pyqtSignal(Doc2VecEmbedder)
    progress = pyqtSignal(int)

    def __init__(self, 
        corpus: Corpus, 
        vector_size: int,
        window: int,
        min_count: int,
        epochs: int,
        max_final_vocab: int
    ):
        super().__init__()
        self.corpus = corpus
        
        workers = multiprocessing.cpu_count() - 2
        if workers < 1:
            workers = 1
        self.model = Doc2Vec(dm=1, dm_mean=1,
            vector_size=vector_size, window=window, epochs=epochs, workers=workers, max_final_vocab=max_final_vocab
        )
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def batch_documents(self, batch_size: int) -> list[TaggedDocument]:
        for index in range(0, len(self.corpus), batch_size):
            documents = [
                TaggedDocument(words=simple_preprocess(doc), tags=[str(i+index)])
                for i, doc in enumerate(self.corpus.documents[index:index+batch_size])
            ]
            yield documents
        return None

    def run(self):
        batch_size = 10
        processed = 0
        last_progress = 0
        for batch in self.batch_documents(len(self.corpus)):
            self.model.build_vocab(batch)

        for batch in self.batch_documents(batch_size):
            self.model.train(batch, total_examples=len(batch), epochs=1)
            processed += len(batch)
            progress = int((processed*100)/len(self.corpus))
            if progress > last_progress:
                self.progress.emit(progress)

        self.result.emit(Doc2VecEmbedder(self.model))

class OWTrainDoc2Vec(widget.OWWidget):
    name = "Train Doc2Vec"
    description = "Trains a new doc2vec gensim model."
    icon = "icons/nlp-doc2vec-train.svg"
    priority = 100

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        embedder = Output("Embedder", Doc2VecEmbedder, auto_summary=False)

    vector_size = settings.Setting(300)
    window_size = settings.Setting(8)
    min_count = settings.Setting(2)
    workers = settings.Setting(4)
    epochs = settings.Setting(40)
    max_final_vocab = settings.Setting(1000000)

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.worker = None
        self.layout_control_area()
    
    def layout_control_area(self):
        # UI setup
        layout = self.controlArea.layout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

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
            self.Outputs.embedder.send(None)
            return

        self.worker = Doc2VecTrainingWorker(
            self.corpus,
            self.vector_size,
            self.window_size,
            self.min_count,
            self.epochs,
            self.max_final_vocab
        )

        self.progressBarInit()
        self.worker.progress.connect(self.progressBarSet)
        self.worker.result.connect(self.finished_training)
        self.setStatusMessage("Training...")
        self.setBlocking(True)
        self.worker.start()

    def finished_training(self, embedder: Doc2VecEmbedder):
        self.progressBarFinished()
        self.setStatusMessage("")
        self.setBlocking(False)
        if embedder:
            self.Outputs.embedder.send(embedder)
            model_dir = os.path.join(data_dir_base(), 'Orange', 'gensim')
            os.makedirs(model_dir, exist_ok=True)
            corpus_name = self.corpus.name or datetime.date.today().strftime("%Y%m%d")
            model_path = os.path.join(model_dir, f"{corpus_name}.{self.vector_size}.bin")
            embedder.save(model_path)
        else:
            self.Outputs.embedder.send(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    corpus = Corpus('book-excerpts')
    WidgetPreview(OWTrainDoc2Vec).run(corpus)