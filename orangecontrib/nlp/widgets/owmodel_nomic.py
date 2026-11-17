from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
from AnyQt.QtWidgets import QLineEdit, QLabel, QSpinBox
from AnyQt.QtCore import Qt
import requests
import faiss
import numpy as np

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class NomicEmbedder(EmbedderModel):

    def __init__(self, host: str = "localhost", port: int = 11434) -> None:
        self.url = f"http://{host}:{port}/api/embeddings"

    def embed(self, language: str, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            response = requests.post(self.url, json={"model": "nomic-embed-text", "prompt": text})
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        return embeddings


class OWNomicEmbedder(OWWidget):
    name = _tr.m[37, "Nomic Embedder"]
    description = _tr.m[57, "Provides the Ollama nomic embedding model"]
    icon = "icons/nlp-nomic.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", NomicEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True
    host = setting("localhost")
    port = setting(11434)

    def __init__(self) -> None:
        super().__init__()
        self.Outputs.embedder.send(NomicEmbedder(self.host, self.port))
        self.layout_control_area()

    def layout_control_area(self) -> None:
        layout_of(self.controlArea).addWidget(QLabel(_tr.m[26, "Host:"]))
        self.host_input = QLineEdit(self.host)
        self.host_input.editingFinished.connect(self.on_host_port_changed)
        layout_of(self.controlArea).addWidget(self.host_input)
        layout_of(self.controlArea).addWidget(QLabel(_tr.m[50, "Port:"]))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(self.port)
        self.port_input.editingFinished.connect(self.on_host_port_changed)
        layout_of(self.controlArea).addWidget(self.port_input)
        layout_of(self.controlArea).setAlignment(Qt.AlignmentFlag.AlignTop)

    def on_host_port_changed(self) -> None:
        self.host = self.host_input.text()
        self.port = self.port_input.value()
        self.Outputs.embedder.send(NomicEmbedder(self.host, self.port))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWNomicEmbedder).run()
