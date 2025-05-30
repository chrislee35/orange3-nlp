from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.widgets.widget import Output, OWWidget
from transformers import AutoTokenizer, AutoModel

# TODO: Make this thread safe
class E5Embedder(EmbedderModel):
    _tokenizer = None
    _model = None

    def __init__(self):
        if E5Embedder._model is None:
            E5Embedder._tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
            E5Embedder._model = AutoModel.from_pretrained("intfloat/e5-small-v2")

    def embed(self, texts: list[str]) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = E5Embedder._model.to(device)
        tokenizer = E5Embedder._tokenizer
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs)
        embeddings = output.last_hidden_state.mean(dim=1)
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return norm_embeddings.cpu().numpy()

class OWE5Embedder(OWWidget):
    name = "E5 Embedder"
    description = "Provides the E5 model"
    icon = "icons/nlp-e5.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", E5Embedder, auto_summary=False)

    want_main_area = False
    want_control_area = False

    def __init__(self):
        super().__init__()
        self.Outputs.embedder.send(E5Embedder())

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWE5Embedder).run()
