from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
from Orange.widgets.settings import Setting
from AnyQt.QtWidgets import (
    QLineEdit, QLabel, QComboBox
)
from AnyQt.QtCore import Qt
import numpy as np
import openai
from typing import get_args

class OpenAIEmbedder(EmbedderModel):
    def __init__(self, api_key, model="text-embedding-3-small"):
        self.api_key = api_key
        self.model = model

    def embed(self, language, texts):
        client = openai.OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            input=texts,
            model=self.model
        )
        return np.array([item.embedding for item in response.data], dtype="float32")

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
    openai_model = Setting("text-embedding-3-small")
 
    def __init__(self):
        super().__init__()
        self.layout_control_area()
        self.update()

    def layout_control_area(self):
        self.controlArea.layout().addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit(self.api_key)
        self.api_key_input.editingFinished.connect(self.on_api_key_changed)
        self.controlArea.layout().addWidget(self.api_key_input)

        embedding_models = get_args(openai.types.EmbeddingModel)
        self.model_combo = QComboBox()
        self.model_combo.addItems(embedding_models)
        self.model_combo.setCurrentText(self.openai_model)
        self.model_combo.currentTextChanged.connect(self.on_change_model)

        self.controlArea.layout().addWidget(QLabel("Embedding Model:"))
        self.controlArea.layout().addWidget(self.model_combo)
        
        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def on_api_key_changed(self):
        self.api_key = self.api_key_input.text()
        self.update()

    def on_change_model(self, val):
        self.openai_model = val
        self.update()

    def update(self):
        if self.api_key and self.openai_model:
            self.Outputs.embedder.send(OpenAIEmbedder(self.api_key, self.openai_model))
        else:
            self.Outputs.embedder.send(None)
            print("None")

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWOpenAIEmbedder).run()
