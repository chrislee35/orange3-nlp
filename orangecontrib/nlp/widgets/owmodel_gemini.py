from Orange.widgets.widget import OWWidget, Output
from AnyQt.QtWidgets import QPushButton
from AnyQt.QtCore import Qt
from orangecontrib.nlp.util.embedder_models import EmbedderModel
from orangecontrib.nlp.util.apikeys import get_api_key
from orangecontrib.nlp.widgets.settings.apikey_preferences import APIKeySettings
import numpy as np
import google.generativeai as genai

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class GeminiEmbedder(EmbedderModel):

    def __init__(self, model: str = "embedding-001") -> None:
        self.model = model
        api_key = get_api_key("gemini")
        genai.configure(api_key=api_key)  # pyright: ignore[reportPrivateImportUsage]

    def embed(self, language: str, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            response = genai.embed_content(  # pyright: ignore[reportPrivateImportUsage]
                model=self.model, content=text, task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        return np.array(embeddings, dtype="float32")


class OWGeminiEmbedder(OWWidget):
    name = _tr.m[22, "Gemini Embedder"]
    description = _tr.m[53, "Provides Google Gemini embeddings via the embedding-001 model"]
    icon = "icons/nlp-gemini.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", GeminiEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True

    def __init__(self) -> None:
        super().__init__()
        self.layout_control_area()
        self.update_embedder()

    def layout_control_area(self) -> None:
        btn = QPushButton(_tr.m[12, "Configure API Keys..."])
        btn.clicked.connect(self.show_apikey_settings)
        layout_of(self.controlArea).addWidget(btn)
        layout_of(self.controlArea).setAlignment(Qt.AlignmentFlag.AlignTop)

    def show_apikey_settings(self) -> None:
        dlg = APIKeySettings(self)
        dlg.show()
        status = dlg.exec()
        if status == 0:
            self.update_embedder()

    def update_embedder(self) -> None:
        if get_api_key("gemini"):
            self.Outputs.embedder.send(GeminiEmbedder())
        else:
            self.Outputs.embedder.send(None)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWGeminiEmbedder).run()
