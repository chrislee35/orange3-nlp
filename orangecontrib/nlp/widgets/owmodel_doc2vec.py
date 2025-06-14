from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.misc.environ import data_dir_base
from Orange.widgets.widget import Output, OWWidget
from Orange.widgets.settings import Setting
from AnyQt.QtWidgets import (
    QComboBox, QLabel, QTextEdit, QPushButton
)
from AnyQt.QtCore import Qt

from gensim.models.doc2vec import Doc2Vec
import numpy as np
import os
from pathlib import Path

class Doc2VecEmbedder(EmbedderModel):
    def __init__(self, model_name):
        self.model_path = os.path.join(data_dir_base(), 'Orange', 'gensim', model_name+'.bin')
        if not os.path.exists(self.model_path):
            raise Exception(f"Model {model_name} not found.")

    def embed(self, language, texts):
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

    model_name = Setting("")
    
    def __init__(self):
        super().__init__()
        self.layout_control_area()
        if self.model_name:
            try:
                self.Outputs.embedder.send(Doc2VecEmbedder(self.model_name))
            except Exception as e:
                self.error_box.setPlainText(str(e))

    def layout_control_area(self):
        models = self.enumerate_models()
        if len(models) > 0 and self.model_name == "":
            self.model_name = models[0]

        self.controlArea.layout().addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(self.model_name)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.controlArea.layout().addWidget(self.model_combo)

        self.error_box = QTextEdit()
        self.error_box.setReadOnly(True)
        self.error_box.setStyleSheet("color: red;")
        self.controlArea.layout().addWidget(QLabel("Model Errors:"))
        self.controlArea.layout().addWidget(self.error_box)

        if len(models) == 0:
            self.error_box.setPlainText("No models were found. Please create them using Train Doc2Vec")

        reload_button = QPushButton("Rescan Models")
        reload_button.clicked.connect(self.rescan_models)
        self.controlArea.layout().addWidget(reload_button)
        
        self.controlArea.layout().setAlignment(Qt.AlignTop)
    
    def on_model_changed(self):
        self.model_name = self.model_combo.currentText()
        if not self.model_name:
            self.Outputs.embedder.send(None)
        else:    
            self.Outputs.embedder.send(Doc2VecEmbedder(self.model_name))

    def enumerate_models(self):
        model_path = os.path.join(data_dir_base(), 'Orange', 'gensim')
        return sorted([f.stem for f in Path(model_path).glob("*.bin")])

    def rescan_models(self):
        models = self.enumerate_models()
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if self.model_name == "" and len(models) > 0:
            self.model_name = models[0]
            self.model_combo.setCurrentText(self.model_name)
        if len(models) > 0:
            self.error_box.setPlainText("")

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDoc2VecEmbedder).run()
