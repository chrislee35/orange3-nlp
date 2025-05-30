
import numpy as np
import os
import urllib.request
import fasttext
from pathlib import Path
from zipfile import ZipFile
import gzip
import shutil

from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
from Orange.widgets.settings import Setting
from AnyQt.QtWidgets import (
    QLineEdit, QLabel
)
from AnyQt.QtCore import Qt


class FastTextEmbedder(EmbedderModel):
    _model = None

    def __init__(self, lang: str = "en", model_dir: str = "fasttext_models"):
        self.lang = lang
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.model_path = os.path.join(model_dir, f"cc.{lang}.300.bin")
        if not os.path.exists(self.model_path):
            self._download_model()

    def embed(self, texts):
        if FastTextEmbedder._model is None:
            FastTextEmbedder._model = fasttext.load_model(self.model_path)
        embeddings = []
        for text in texts:
            words = text.strip().split()
            vectors = [FastTextEmbedder._model.get_word_vector(word) for word in words]
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(FastTextEmbedder._model.get_dimension(), dtype="float32")
            embeddings.append(doc_vector)
        return np.array(embeddings, dtype="float32")


    def _download_model(self):
        print(f"Downloading FastText model for language '{self.lang}'...")

        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{self.lang}.300.bin.gz"
        gz_path = self.model_path + ".gz"

        try:
            urllib.request.urlretrieve(url, gz_path)
            print("Download complete. Extracting...")

            with gzip.open(gz_path, 'rb') as f_in, open(self.model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            os.remove(gz_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download FastText model for '{self.lang}': {e}")

class OWFastTextEmbedder(OWWidget):
    name = "FastText Embedder"
    description = "Provides the FastText embedding model"
    icon = "icons/nlp-fasttext.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", FastTextEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True

    lang = Setting("en")
    model_dir = Setting("fasttext_models")
 
    def __init__(self):
        super().__init__()
        self.Outputs.embedder.send(FastTextEmbedder(self.lang))
        self.layout_control_area()

    def layout_control_area(self):
        self.controlArea.layout().addWidget(QLabel("Language:"))
        self.lang_input = QLineEdit(self.lang)
        self.lang_input.editingFinished.connect(self.on_lang_changed)
        self.controlArea.layout().addWidget(self.lang_input)
        self.controlArea.layout().setAlignment(Qt.AlignTop)     

    def on_lang_changed(self):
        self.lang = self.lang_input.text()
        self.Outputs.embedder.send(FastTextEmbedder(self.lang))

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWFastTextEmbedder).run()
