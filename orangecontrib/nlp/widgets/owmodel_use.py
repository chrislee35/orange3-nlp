from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
import tensorflow_hub as hub

class USEEmbedder(EmbedderModel):
    def embed(self, language, texts):
        if not hasattr(self, "_model"):
            self._model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        embeddings = self._model(texts)
        return embeddings.numpy()

class OWUSEEmbedder(OWWidget):
    name = "USE Embedder"
    description = "Provides the sUSE embedder model"
    icon = "icons/nlp-USE.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", USEEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = False

    def __init__(self):
        super().__init__()
        self.Outputs.embedder.send(USEEmbedder())

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWUSEEmbedder).run()
