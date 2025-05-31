from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
from Orange.widgets.settings import Setting
from AnyQt.QtWidgets import (
    QLineEdit, QLabel, QSpinBox
)
from AnyQt.QtCore import Qt
import requests
import faiss
import numpy as np

class NomicEmbedder(EmbedderModel):
    def __init__(self, host: str = 'localhost', port: int = 11434):
        self.url = f"http://{host}:{port}/api/embeddings"

    def embed(self, language, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            response = requests.post(self.url, json={
                "model": "nomic-embed-text",
                "prompt": text
            })
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        return embeddings

class OWNomicEmbedder(OWWidget):
    name = "Nomic Embedder"
    description = "Provides the Ollama nomic embedding model"
    icon = "icons/nlp-nomic.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", NomicEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True

    host = Setting("localhost")
    port = Setting(11434)
 
    def __init__(self):
        super().__init__()
        self.Outputs.embedder.send(NomicEmbedder(self.host, self.port))
        self.layout_control_area()

    def layout_control_area(self):
        self.controlArea.layout().addWidget(QLabel("Host:"))
        self.host_input = QLineEdit(self.host)
        self.host_input.editingFinished.connect(self.on_host_port_changed)
        self.controlArea.layout().addWidget(self.host_input)

        self.controlArea.layout().addWidget(QLabel("Port:"))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(self.port)
        self.port_input.editingFinished.connect(self.on_host_port_changed)
        self.controlArea.layout().addWidget(self.port_input)
        self.controlArea.layout().setAlignment(Qt.AlignTop)     

    def on_host_port_changed(self):
        self.host = self.host_input.text()
        self.port = self.port_input.value()
        self.Outputs.embedder.send(NomicEmbedder(self.host, self.port))

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWNomicEmbedder).run()
