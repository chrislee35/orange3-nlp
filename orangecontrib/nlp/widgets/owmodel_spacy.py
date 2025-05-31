from Orange.widgets.widget import Output, OWWidget
from orangecontrib.nlp.util.embedder_models import EmbedderModel
from orangecontrib.nlp.util.spacy_downloader import SpaCyDownloader
import spacy
import threading
import numpy as np

class SpacyEmbedder(EmbedderModel):
    _thread_local = threading.local()

    def _get_model(self, language):
        if not hasattr(self._thread_local, "model"):
            self._thread_local.model = {}
            
        if language not in self._thread_local.model:
            model_name = SpaCyDownloader.model_name(language)
            try:
                self._thread_local.model[language] = spacy.load(model_name)
            except Exception as e:
                SpaCyDownloader.download(model_name)
                self._thread_local.model[language] = spacy.load(model_name)
        return self._thread_local.model[language]

    def embed(self, language, texts: list[str]) -> np.ndarray:
        model = self._get_model(language)
        return np.array([model(text).vector for text in texts], dtype="float32")

class OWSpacyEmbedder(OWWidget):
    name = "spaCy Embedder"
    description = "Provides the spaCy embedding model"
    icon = "icons/nlp-spacy.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", SpacyEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = False

    def __init__(self):
        super().__init__()
        self.Outputs.embedder.send(SpacyEmbedder())

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSpacyEmbedder).run()
