import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import requests
from orangecontrib.nlp.util.spacy_downloader import SpaCyDownloader

class EmbedderModel:
    _spacy_model = None
    _sbert_model = None
    _hf_tokenizer = None
    _hf_model = None

    def __init__(self, embedder_name):
        self.embedder = self.get_embedder(embedder_name)

    def embed_texts(self, texts: list[str]):
        return self.embedder(texts)

    @staticmethod
    def get_embedder(name):
        if name == "spacy":
            SpaCyDownloader.download("en_core_web_md")
            if EmbedderFactory._spacy_model is None:
                EmbedderFactory._spacy_model = spacy.load("en_core_web_md")
            def embed(texts):
                return np.array([EmbedderFactory._spacy_model(t).vector for t in texts], dtype="float32")
            return embed
        elif name == "sentence-transformers":
            if EmbedderFactory._sbert_model is None:
                EmbedderFactory._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            return lambda texts: EmbedderFactory._sbert_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        elif name == "e5-small-v2":
            if EmbedderFactory._hf_model is None:
                EmbedderFactory._hf_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
                EmbedderFactory._hf_model = AutoModel.from_pretrained("intfloat/e5-small-v2")
            def embed(texts):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = EmbedderFactory._hf_model.to(device)
                inputs = EmbedderFactory._hf_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    model_output = model(**inputs)
                embeddings = model_output.last_hidden_state.mean(dim=1)
                norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return norm_embeddings.cpu().numpy()
            return embed
        elif name == "nomic-embed-text":
            def embed(texts):
                url = "http://localhost:11434/api/embeddings"
                embeddings = []
                for text in texts:
                    response = requests.post(url, json={"model": "nomic-embed-text", "prompt": text})
                    response.raise_for_status()
                    data = response.json()
                    embeddings.append(data["embedding"])
                embeddings = np.array(embeddings, dtype="float32")
                faiss.normalize_L2(embeddings)
                return embeddings
            return embed
        else:
            raise ValueError("Unknown embedder")

class SpacyEmbedder(EmbedderModel):
    def __init__(self):
        super().__init__("spacy")

class SBERTEmbedder(EmbedderModel):
    def __init__(self):
        super().__init__("sentence-transformers")

class E5Embedder(EmbedderModel):
    def __init__(self):
        super().__init__("e5-small-v2")

class NomicEmbedder(EmbedderModel):
    def __init__(self):
        super().__init__("nomic-embed-text")
