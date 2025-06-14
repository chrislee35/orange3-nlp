from Orange.widgets.widget import OWWidget, Output
from AnyQt.QtWidgets import QPushButton
from AnyQt.QtCore import Qt
from orangecontrib.nlp.util.embedder_models import EmbedderModel
from orangecontrib.nlp.util.apikeys import get_api_key
from orangecontrib.nlp.widgets import APIKeySettings

import numpy as np
import google.generativeai as genai

class GeminiEmbedder(EmbedderModel):
    def __init__(self, model: str = "embedding-001"):
        self.model = model
        api_key = get_api_key("gemini")
        genai.configure(api_key=api_key)

    def embed(self, language, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        # If multiple texts are embedded separately, repeat this in a loop
        return np.array(embeddings, dtype="float32")

class OWGeminiEmbedder(OWWidget):
    name = "Gemini Embedder"
    description = "Provides Google Gemini embeddings via the embedding-001 model"
    icon = "icons/nlp-gemini.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", GeminiEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = True

    def __init__(self):
        super().__init__()
        self.layout_control_area()
        self.update()

    def layout_control_area(self):
        btn = QPushButton("Configure API Keys...")
        btn.clicked.connect(self.show_apikey_settings)
        self.controlArea.layout().addWidget(btn)
        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def show_apikey_settings(self):
        dlg = APIKeySettings(self)
        dlg.show()
        status = dlg.exec()
        if status == 0:
            self.update()

    def update(self):
        if get_api_key("gemini"):
            self.Outputs.embedder.send(GeminiEmbedder())
        else:
            self.Outputs.embedder.send(None)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWGeminiEmbedder).run()

