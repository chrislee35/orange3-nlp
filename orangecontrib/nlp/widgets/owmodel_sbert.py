from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
from sentence_transformers import SentenceTransformer
import numpy as np

from orangecanvas.localization import Translator

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class SBERTEmbedder(EmbedderModel):
    _model: SentenceTransformer | None = None

    def __init__(self) -> None:
        if SBERTEmbedder._model is None:
            SBERTEmbedder._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, language: str, texts: list[str]) -> np.ndarray:
        model = SBERTEmbedder._model
        assert model is not None
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class OWSBERTEmbedder(OWWidget):
    name = _tr.m[65, "SBERT Embedder"]
    description = _tr.m[60, "Provides the sentence-transformers embedder model"]
    icon = "icons/nlp-sbert.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", SBERTEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = False

    def __init__(self) -> None:
        super().__init__()
        self.Outputs.embedder.send(SBERTEmbedder())


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSBERTEmbedder).run()
