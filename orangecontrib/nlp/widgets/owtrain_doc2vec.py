from collections.abc import Iterator
from typing import Any

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget
from Orange.widgets.widget import Input, Output
from orangecontrib.text.corpus import Corpus
from orangecontrib.nlp.util.embedder_models import EmbedderModel
from orangecontrib.nlp.util.paths import orange_data_dir
import numpy as np
import multiprocessing
import os
import datetime

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class Doc2VecEmbedder(EmbedderModel):

    def __init__(self, model: Doc2Vec) -> None:
        self.model = model

    def embed_doc2vec(self, model: Any, text: str) -> np.ndarray:
        tokens = simple_preprocess(text)
        vector = model.infer_vector(tokens)
        return np.array(vector, dtype=np.float32)

    def embed(self, language: str, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_doc2vec(self.model, text) for text in texts])

    def save(self, *args: Any, **kwargs: Any) -> None:
        print(*args)
        self.model.save(*args, **kwargs)


class Doc2VecTrainingWorker(QThread):
    result = pyqtSignal(Doc2VecEmbedder)
    progress = pyqtSignal(int)

    def __init__(
        self,
        corpus: Corpus,
        vector_size: int,
        window: int,
        min_count: int,
        epochs: int,
        max_final_vocab: int,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        workers = multiprocessing.cpu_count() - 2
        if workers < 1:
            workers = 1
        self.model = Doc2Vec(
            dm=1,
            dm_mean=1,
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            workers=workers,
            max_final_vocab=max_final_vocab,
        )
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def batch_documents(self, batch_size: int) -> Iterator[list[TaggedDocument]]:
        for index in range(0, len(self.corpus), batch_size):
            documents = [
                TaggedDocument(words=simple_preprocess(doc), tags=[str(i + index)])
                for i, doc in enumerate(self.corpus.documents[index : index + batch_size])
            ]
            yield documents

    def run(self) -> None:
        batch_size = 10
        processed = 0
        last_progress = 0
        for batch in self.batch_documents(len(self.corpus)):
            self.model.build_vocab(batch)
        for batch in self.batch_documents(batch_size):
            self.model.train(batch, total_examples=len(batch), epochs=1)
            processed += len(batch)
            progress = int(processed * 100 / len(self.corpus))
            if progress > last_progress:
                self.progress.emit(progress)
        self.result.emit(Doc2VecEmbedder(self.model))


class OWTrainDoc2Vec(widget.OWWidget):
    name = _tr.m[77, "Train Doc2Vec"]
    description = _tr.m[78, "Trains a new doc2vec gensim model."]
    icon = "icons/nlp-doc2vec-train.svg"
    priority = 100

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        embedder = Output("Embedder", Doc2VecEmbedder, auto_summary=False)

    vector_size = setting(300)
    window_size = setting(8)
    min_count = setting(2)
    workers = setting(4)
    epochs = setting(40)
    max_final_vocab = setting(1000000)
    want_main_area = False

    def __init__(self) -> None:
        super().__init__()
        self.corpus = None
        self.worker = None
        self.layout_control_area()

    def layout_control_area(self) -> None:
        layout = layout_of(self.controlArea)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    @Inputs.data
    def set_data(self, data: Corpus) -> None:
        self.corpus = data
        self.apply()

    def on_chunk_size_change(self, val: int) -> None:
        self.chunk_size = int(val)
        self.apply()

    def apply(self) -> None:
        if not self.corpus:
            self.Outputs.embedder.send(None)
            return
        self.worker = Doc2VecTrainingWorker(
            self.corpus,
            self.vector_size,
            self.window_size,
            self.min_count,
            self.epochs,
            self.max_final_vocab,
        )
        self.progressBarInit()
        self.worker.progress.connect(self.progressBarSet)
        self.worker.result.connect(self.finished_training)
        self.setStatusMessage("Training...")
        self.setBlocking(True)
        self.worker.start()

    def finished_training(self, embedder: Doc2VecEmbedder) -> None:
        self.progressBarFinished()
        self.setStatusMessage("")
        self.setBlocking(False)
        if embedder:
            self.Outputs.embedder.send(embedder)
            model_dir = orange_data_dir("gensim")
            os.makedirs(model_dir, exist_ok=True)
            corpus_name = (self.corpus.name if self.corpus is not None else None) or datetime.date.today().strftime("%Y%m%d")
            model_path = os.path.join(model_dir, f"{corpus_name}.{self.vector_size}.bin")
            embedder.save(model_path)
        else:
            self.Outputs.embedder.send(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    corpus = Corpus("book-excerpts")
    WidgetPreview(OWTrainDoc2Vec).run(corpus)
