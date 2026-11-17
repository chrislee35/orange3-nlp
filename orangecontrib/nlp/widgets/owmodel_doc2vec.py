from typing import Any

from orangecontrib.nlp.util.embedder_models import EmbedderModel
from orangecontrib.nlp.util.paths import orange_data_dir
from Orange.widgets.widget import Output, OWWidget
from AnyQt.QtWidgets import QComboBox, QLabel, QTextEdit, QPushButton
from AnyQt.QtCore import Qt
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import os
from pathlib import Path

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class Doc2VecEmbedder(EmbedderModel):

    def __init__(self, model_name: str) -> None:
        self.model_path = orange_data_dir("gensim", model_name + ".bin")
        if not os.path.exists(self.model_path):
            raise Exception(f"Model {model_name} not found.")

    def embed(self, language: str, texts: list[str]) -> np.ndarray:
        if not hasattr(self, "_model"):
            self._model: Any = Doc2Vec.load(self.model_path)
        return np.array([self._model.infer_vector(text.split()) for text in texts], dtype="float32")


class OWDoc2VecEmbedder(OWWidget):
    name = _tr.m[13, "Doc2Vec Embedder"]
    description = _tr.m[56, "Provides the Gensim Doc2Vec model"]
    icon = "icons/nlp-doc2vec.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", Doc2VecEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True
    model_name = setting("")

    def __init__(self) -> None:
        super().__init__()
        self.layout_control_area()
        if self.model_name:
            try:
                self.Outputs.embedder.send(Doc2VecEmbedder(self.model_name))
            except Exception as e:
                self.error_box.setPlainText(str(e))

    def layout_control_area(self) -> None:
        models = self.enumerate_models()
        if len(models) > 0 and self.model_name == "":
            self.model_name = models[0]
        layout_of(self.controlArea).addWidget(QLabel(_tr.m[30, "Model:"]))
        self.model_combo = QComboBox()
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(self.model_name)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        layout_of(self.controlArea).addWidget(self.model_combo)
        self.error_box = QTextEdit()
        self.error_box.setReadOnly(True)
        self.error_box.setStyleSheet("color: red;")
        layout_of(self.controlArea).addWidget(QLabel(_tr.m[29, "Model Errors:"]))
        layout_of(self.controlArea).addWidget(self.error_box)
        if len(models) == 0:
            self.error_box.setPlainText(_tr.m[34, "No models were found. Please create them using Train Doc2Vec"])
        reload_button = QPushButton(_tr.m[64, "Rescan Models"])
        reload_button.clicked.connect(self.rescan_models)
        layout_of(self.controlArea).addWidget(reload_button)
        layout_of(self.controlArea).setAlignment(Qt.AlignmentFlag.AlignTop)

    def on_model_changed(self) -> None:
        self.model_name = self.model_combo.currentText()
        if not self.model_name:
            self.Outputs.embedder.send(None)
        else:
            self.Outputs.embedder.send(Doc2VecEmbedder(self.model_name))

    def enumerate_models(self) -> list[str]:
        model_path = orange_data_dir("gensim")
        return sorted([f.stem for f in Path(model_path).glob("*.bin")])

    def rescan_models(self) -> None:
        models = self.enumerate_models()
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if self.model_name == "" and len(models) > 0:
            self.model_name = models[0]
            self.model_combo.setCurrentText(self.model_name)
        if len(models) > 0:
            self.error_box.setPlainText(_tr.m[0, ""])


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDoc2VecEmbedder).run()
