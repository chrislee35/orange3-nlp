
import numpy as np
import os
import urllib.request
import fasttext
from pathlib import Path
from zipfile import ZipFile
import gzip
import shutil

from orangecontrib.nlp.util.embedder_models import EmbedderModel
from Orange.misc.environ import data_dir_base
from Orange.widgets.widget import Output, OWWidget
from Orange.widgets.settings import Setting
from AnyQt.QtWidgets import (
    QLineEdit, QLabel
)
from AnyQt.QtCore import Qt


class FastTextEmbedder(EmbedderModel):
    _model = None
    AVAILABLE_LANGUAGES = [
        "af", "sq", "als", "am", "ar", "an", "hy", "as", "ast", "az", "ba", "eu", "bar", "be", "bn", "bh", "bpy", "bs", 
        "br", "bg", "my", "ca", "ceb", "bcl", "ce", "zh", "cv", "co", "hr", "cs", "da", "dv", "nl", "pa", "arz", "eml", 
        "en", "myv", "eo", "et", "hif", "fi", "fr", "gl", "ka", "de", "gom", "el", "gu", "ht", "he", "mrj", "hi", "hu", 
        "is", "io", "ilo", "id", "ia", "ga", "it", "ja", "jv", "kn", "pam", "kk", "km", "ky", "ko", "ku", "ckb", "la", 
        "lv", "li", "lt", "lmo", "nds", "lb", "mk", "mai", "mg", "ms", "ml", "mt", "gv", "mr", "mzn", "mhr", "min", 
        "xmf", "mwl", "mn", "nah", "nap", "ne", "new", "frr", "nso", "no", "nn", "oc", "or", "os", "pfl", "ps", "fa", 
        "pms", "pl", "pt", "qu", "ro", "rm", "ru", "sah", "sa", "sc", "sco", "gd", "sr", "sh", "scn", "sd", "si", "sk", 
        "sl", "so", "azb", "es", "su", "sw", "sv", "tl", "tg", "ta", "tt", "te", "th", "bo", "tr", "tk", "uk", "hsb", 
        "ur", "ug", "uz", "vec", "vi", "vo", "wa", "war", "cy", "vls", "fy", "pnb", "yi", "yo", "diq", "zea"
    ]

    def __init__(self):
        self.model_dir = os.path.join(data_dir_base(), 'Orange', 'fasttext')
        os.makedirs(self.model_dir, exist_ok=True)

    def embed(self, language, texts):
        language = language or "en"
        if language not in FastTextEmbedder.AVAILABLE_LANGUAGES:
            raise Exception(f"{language} is unavailable from https://fasttext.cc/docs/en/crawl-vectors.html")

        model_path = os.path.join(self.model_dir, f"cc.{language}.300.bin")
        if not os.path.exists(model_path):
            self._download_model(language, model_path)

        if FastTextEmbedder._model is None:
            FastTextEmbedder._model = fasttext.load_model(model_path)
        embeddings = []
        for text in texts:
            words = text.strip().split()
            vectors = [FastTextEmbedder._model.get_word_vector(word) for word in words]
            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(FastTextEmbedder._model.get_dimension(), dtype="float32")
            embeddings.append(doc_vector)
        return np.array(embeddings, dtype="float32")


    def _download_model(self, language, model_path):
        print(f"Downloading FastText model for language '{language}'...")

        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.bin.gz"
        gz_path = f"{model_path}.gz"

        try:
            urllib.request.urlretrieve(url, gz_path)
            print("Download complete. Extracting...")

            with gzip.open(gz_path, 'rb') as f_in, open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            os.remove(gz_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download FastText model for '{language}': {e}")

class OWFastTextEmbedder(OWWidget):
    name = "FastText Embedder"
    description = "Provides the FastText embedding model"
    icon = "icons/nlp-fasttext.svg"
    priority = 150

    class Outputs:
        embedder = Output("Embedder", FastTextEmbedder, auto_summary=False)

    want_main_area = False
    want_control_area = False
 
    def __init__(self):
        super().__init__()
        self.Outputs.embedder.send(FastTextEmbedder())

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWFastTextEmbedder).run()
