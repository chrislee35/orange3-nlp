from Orange.widgets.widget import Output, OWWidget
from AnyQt.QtWidgets import QLabel, QComboBox, QPushButton
from AnyQt.QtCore import Qt
from orangecontrib.nlp.util.embedder_models import EmbedderModel
from orangecontrib.nlp.util.apikeys import get_api_key
from orangecontrib.nlp.widgets.settings.apikey_preferences import APIKeySettings
import numpy as np
import openai
from typing import get_args

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class OpenAIEmbedder(EmbedderModel):

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.api_key = get_api_key("openai")
        self.model = model

    def embed(self, language: str, texts: list[str]) -> np.ndarray:
        client = openai.OpenAI(api_key=self.api_key)
        response = client.embeddings.create(input=texts, model=self.model)
        return np.array([item.embedding for item in response.data], dtype="float32")


class OWOpenAIEmbedder(OWWidget):
    name = _tr.m[44, "OpenAI Embedder"]
    description = _tr.m[58, "Provides the Openai embedding model: text-embedding-ada-002"]
    icon = "icons/nlp-openai.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", OpenAIEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True
    openai_model = setting("text-embedding-3-small")

    def __init__(self) -> None:
        super().__init__()
        self.layout_control_area()
        self.update_embedder()

    def layout_control_area(self) -> None:
        btn = QPushButton(_tr.m[12, "Configure API Keys..."])
        btn.clicked.connect(self.show_apikey_settings)
        layout_of(self.controlArea).addWidget(btn)
        embedding_models = get_args(openai.types.EmbeddingModel)
        self.model_combo = QComboBox()
        self.model_combo.addItems(embedding_models)
        self.model_combo.setCurrentText(self.openai_model)
        self.model_combo.currentTextChanged.connect(self.on_change_model)
        layout_of(self.controlArea).addWidget(QLabel(_tr.m[15, "Embedding Model:"]))
        layout_of(self.controlArea).addWidget(self.model_combo)
        layout_of(self.controlArea).setAlignment(Qt.AlignmentFlag.AlignTop)

    def show_apikey_settings(self) -> None:
        dlg = APIKeySettings(self)
        dlg.show()
        status = dlg.exec()
        if status == 0:
            self.update_embedder()

    def on_change_model(self, val: str) -> None:
        self.openai_model = val
        self.update_embedder()

    def update_embedder(self) -> None:
        api_key = get_api_key("openai")
        if api_key and self.openai_model:
            self.Outputs.embedder.send(OpenAIEmbedder(self.openai_model))
        else:
            self.Outputs.embedder.send(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWOpenAIEmbedder).run()
