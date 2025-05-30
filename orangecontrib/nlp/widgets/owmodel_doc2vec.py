from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
from Orange.widgets.settings import Setting
from AnyQt.QtWidgets import (
    QLineEdit, QLabel
)
from AnyQt.QtCore import Qt

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import os

class Doc2VecEmbedder(EmbedderModel):
    def __init__(self, model_path):
        self.model_path = model_path

    def embed(texts):
        if not hasattr(self, "_model"):
            self._model = Doc2Vec.load(self.model_path)
        return np.array([self._model.infer_vector(text.split()) for text in texts], dtype="float32")

class OWDoc2VecEmbedder(OWWidget):
    name = "Doc2Vec Embedder"
    description = "Provides the Gensim Doc2Vec model"
    icon = "icons/nlp-doc2vec.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", Doc2VecEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True

    model_path = Setting(f"{os.environ['HOME']}/.gensim")

    def __init__(self):
        super().__init__()
        self.Outputs.embedder.send(Doc2VecEmbedder(self.model_path))
        self.layout_control_area()

    def layout_control_area(self):
        self.controlArea.layout().addWidget(QLabel("Model Path:"))
        self.model_path_input = QLineEdit(self.model_path)
        self.model_path_input.editingFinished.connect(self.on_model_path_changed)
        self.controlArea.layout().addWidget(self.model_path_input)
        self.controlArea.layout().setAlignment(Qt.AlignTop)     

    def on_model_path_changed(self):
        self.model_path = self.model_path_input.text()
        self.Outputs.embedder.send(Doc2VecEmbedder(self.model_path))    

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDoc2VecEmbedder).run()
