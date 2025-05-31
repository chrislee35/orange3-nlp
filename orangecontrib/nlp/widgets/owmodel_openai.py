from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
from Orange.widgets.settings import Setting
from AnyQt.QtWidgets import (
    QLineEdit, QLabel
)
from AnyQt.QtCore import Qt
import numpy as np
import openai

class OpenAIEmbedder(EmbedderModel):
    def __init__(self, api_key):
        self.api_key = api_key

    def embed(self, language, texts):
        openai.api_key = self.api_key
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return np.array([item["embedding"] for item in response["data"]], dtype="float32")

class OWOpenAIEmbedder(OWWidget):
    name = "OpenAI Embedder"
    description = "Provides the Openai embedding model: text-embedding-ada-002"
    icon = "icons/nlp-openai.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", OpenAIEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True

    api_key = Setting("")
 
    def __init__(self):
        super().__init__()
        self.Outputs.embedder.send(OpenAIEmbedder(self.api_key))
        self.layout_control_area()

    def layout_control_area(self):
        self.controlArea.layout().addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit(self.api_key)
        self.api_key_input.editingFinished.connect(self.on_api_key_changed)
        self.controlArea.layout().addWidget(self.api_key_input)
        self.controlArea.layout().setAlignment(Qt.AlignTop)     

    def on_api_key_changed(self):
        self.api_key = self.api_key_input.text()
        self.Outputs.embedder.send(OpenAIEmbedder(self.api_key))

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWOpenAIEmbedder).run()
