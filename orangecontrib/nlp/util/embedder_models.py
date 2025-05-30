import abc
import faiss
import numpy as np
import openai
import requests
import tensorflow as tf
import tensorflow_hub as hub
import threading
import torch
import torch.nn.functional as F




import os
import urllib.request
import fasttext
from pathlib import Path
from zipfile import ZipFile
import gzip
import shutil

class EmbedderModel(abc.ABC):
    """Abstract base class for embedders."""
    
    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        pass





class FastTextEmbedder(EmbedderModel):
    _model = None

    def __init__(self, lang: str = "en", model_dir: str = "fasttext_models"):
        self.lang = lang
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.model_path = os.path.join(model_dir, f"cc.{lang}.300.bin")
        if not os.path.exists(self.model_path):
            self._download_model()

    def get_embedder(self, name, host=None, port=None):
        if FastTextEmbedder._model is None:
            FastTextEmbedder._model = fasttext.load_model(self.model_path)

        def embed(texts):
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

        return embed

    def _download_model(self):
        print(f"Downloading FastText model for language '{self.lang}'...")

        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{self.lang}.300.bin.gz"
        gz_path = self.model_path + ".gz"

        try:
            urllib.request.urlretrieve(url, gz_path)
            print("Download complete. Extracting...")

            with gzip.open(gz_path, 'rb') as f_in, open(self.model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            os.remove(gz_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download FastText model for '{self.lang}': {e}")
