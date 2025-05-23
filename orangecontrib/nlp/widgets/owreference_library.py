from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox,
    QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QWidget
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import Domain, StringVariable, Table
from orangecontrib.text.corpus import Corpus
import numpy as np
import uuid
import faiss
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import requests
from orangecontrib.nlp.util.spacy_downloader import SpaCyDownloader

from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbedderFactory:
    _spacy_model = None
    _sbert_model = None
    _hf_tokenizer = None
    _hf_model = None

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

class VectorDB(QThread):
    result = pyqtSignal(object)  # emits the built VectorDB
    progress = pyqtSignal(int)     # emits progress (0-100)

    def __init__(self, texts, metadata, embed_func, chunk_size: int=256):
        super().__init__()
        self.texts = texts
        self.metadata = metadata
        self.chunks = []
        self.metadata_idx = []
        self.index = None
        self.embed_func = embed_func
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap  = int(chunk_size * 0.1)
        )

    def calculate_progress(self, part, whole, startpercent: int = 0, endpercent: int = 100) -> int:
        return int((part / whole)*(endpercent - startpercent) + startpercent)

    def run(self):
        self.metadata_idx = []
        total = len(self.texts)
        last_progress = 0
        for idx, text in enumerate(self.texts):
            chunks = self.text_splitter.split_text(text)
            self.chunks.extend(chunks)
            self.metadata_idx.extend([idx] * len(chunks))
            progress = self.calculate_progress(idx+1, total, 0, 20)
            if progress > last_progress:
                self.progress.emit(progress)
                last_progress = progress
        
        batch_size = 32  # You can adjust this based on memory/performance
        vectors = []

        total_batches = (len(self.chunks) + batch_size - 1) // batch_size
        idx = 0
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            vecs = self.embed_func(batch)
            vectors.append(vecs)
            progress = self.calculate_progress(idx+1, total_batches, 20, 98)
            idx += 1
            if progress > last_progress:
                self.progress.emit(progress)
                last_progress = progress

        vectors = np.vstack(vectors)
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        batch_size = 256
        num_vectors = vectors.shape[0]
        total_batches = (num_vectors + batch_size - 1) // batch_size
        idx = 0
        for i in range(0, num_vectors, batch_size):
            self.index.add(vectors[i:i + batch_size])
            progress = self.calculate_progress(idx+1, total_batches, 98, 100)
            idx += 1
            if progress > last_progress:
                self.progress.emit(progress)
                last_progress = progress

        self.result.emit(self.index)

    def search(self, query, top_k=5):
        if self.index is None:
            return []
        query_vec = self.embed_func([query])
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, top_k)
        return [(self.texts[i], self.metadata[self.metadata_idx[i]], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.texts)]

class SearchWorker(QThread):
    result = pyqtSignal(list)
    progress = pyqtSignal(int)

    def __init__(self, query, vector_db, top_k):
        super().__init__()
        self.query = query
        self.vector_db = vector_db
        self.top_k = top_k
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        self.progress.emit(10)
        results = self.vector_db.search(self.query, top_k=self.top_k)
        self.progress.emit(100)
        if not self._cancelled:
            self.result.emit(results)


class OWReferenceLibrary(widget.OWWidget):
    name = "Reference Library"
    description = "Stores documents in a vector database and retrieves references."
    icon = "icons/nlp-reference.svg"
    priority = 150

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Corpus", Corpus)

    embedder = settings.Setting("sentence-transformers")
    max_excerpts = settings.Setting(5)
    threshold = settings.Setting(0.0)
    chunk_size = settings.Setting(256)
    query = settings.Setting("")
    ollama_host = settings.Setting("localhost")
    ollama_port = settings.Setting("11434")

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.vector_db = None
        self.worker = None

        self.layout_control_area()
        self.layout_main_area()

    def layout_control_area(self):
        self.controlArea.layout().addWidget(QLabel("Embedder:"))
        self.embedder_combo = QComboBox()
        self.embedder_combo.addItems([
            "sentence-transformers", "e5-small-v2", "nomic-embed-text", "spacy"
        ])
        self.embedder_combo.setCurrentText(self.embedder)
        self.embedder_combo.currentTextChanged.connect(self.on_embedder_change)
        self.controlArea.layout().addWidget(self.embedder_combo)

        max_excerpts_label = QLabel("Max excerpts:")
        self.controlArea.layout().addWidget(max_excerpts_label)
        self.max_excerpts_spin = QSpinBox()
        self.max_excerpts_spin.setRange(1, 100)
        self.max_excerpts_spin.setValue(self.max_excerpts)
        self.max_excerpts_spin.valueChanged.connect(self.on_max_excerpts_change)
        self.controlArea.layout().addWidget(self.max_excerpts_spin)

        threshold_label = QLabel("Matching threshold:")
        self.controlArea.layout().addWidget(threshold_label)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(self.on_threshold_change)
        self.controlArea.layout().addWidget(self.threshold_spin)

        chunk_size_label = QLabel("Chunk size:")
        chunk_size_label.setToolTip("Controls the size of the excerpts that are indexed and returned.")
        self.controlArea.layout().addWidget(chunk_size_label)
        self.chunk_size_combo = QComboBox()
        self.chunk_size_combo.addItems(["128", "256", "512", "1024"])
        self.chunk_size_combo.setCurrentText(str(self.chunk_size))
        self.chunk_size_combo.currentTextChanged.connect(self.on_chunk_size_change)
        self.controlArea.layout().addWidget(self.chunk_size_combo)

        # Ollama host/port config panel (initially hidden)
        self.ollama_panel = QWidget()
        ollama_layout = QVBoxLayout()
        self.ollama_panel.setLayout(ollama_layout)

        self.host_input = QLineEdit(self.ollama_host)
        self.port_input = QLineEdit(self.ollama_port)
        self.host_input.setPlaceholderText("Ollama Host")
        self.port_input.setPlaceholderText("Ollama Port")

        ollama_layout.addWidget(QLabel("Ollama Host:"))
        ollama_layout.addWidget(self.host_input)
        ollama_layout.addWidget(QLabel("Ollama Port:"))
        ollama_layout.addWidget(self.port_input)

        self.controlArea.layout().addWidget(self.ollama_panel)
        self.ollama_panel.setVisible(self.embedder_combo.currentText() == "nomic-embed-text")

        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def layout_main_area(self):
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter query here")
        self.mainArea.layout().addWidget(self.query_input)

        buttons_layout = QHBoxLayout()
        self.search_button = QPushButton("Find References")
        self.search_button.clicked.connect(self.find_references)
        buttons_layout.addWidget(self.search_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_worker)
        buttons_layout.addWidget(self.stop_button)
        self.mainArea.layout().addLayout(buttons_layout)

        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.mainArea.layout().addWidget(self.results_display)

    def on_embedder_change(self, text):
        self.embedder = text
        self.ollama_panel.setVisible(text == "nomic-embed-text")
        self.build_vector_db()

    def on_max_excerpts_change(self, val):
        self.max_excerpts = val

    def on_threshold_change(self, val):
        self.threshold = val

    def on_chunk_size_change(self, val):
        self.chunk_size = int(val)
        self.build_vector_db()

    @Inputs.data
    def set_data(self, data):
        self.corpus = data
        self.build_vector_db()

    def build_vector_db(self):
        if not (self.corpus and self.embedder):
            return
        embed_func = EmbedderFactory.get_embedder(self.embedder)
        texts = self.corpus.documents
        metadata = list(range(len(self.corpus)))
        self.vector_db = VectorDB(texts, metadata, embed_func, self.chunk_size)
        self.worker = self.vector_db
        self.progressBarInit()
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.finish_vector_db_indexing)
        self.worker.start()

    def update_progress(self, value):
        self.progressBarSet(value)

    def finish_vector_db_indexing(self, index: object):
        self.progressBarFinished()

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            self.progressBarInit()

    def find_references(self):
        if not self.corpus or not self.query_input.text():
            return

        self.stop_worker()

        self.progressBarInit()
        self.query = self.query_input.text()
        self.worker = SearchWorker(self.query, self.vector_db, top_k=self.max_excerpts)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.display_results)
        self.worker.start()
        self.save_ollama_config()

    def save_ollama_config(self):
        self.ollama_host = self.host_input.text()
        self.ollama_port = self.port_input.text()
        
    def display_results(self, results):
        excerpt_var = StringVariable("excerpt")
        score_var = StringVariable("score")
        metas = [excerpt_var, score_var]

        new_rows = []
        for text, meta_idx, score in results:
            if score < self.threshold:
                continue
            #original = self.corpus[int(meta_idx)]
            new_metas = [text, f"{score:.4f}"]
            new_rows.append(new_metas)

        if not new_rows:
            self.results_display.setPlainText("No results above the threshold.")
            return

        domain = Domain([], metas=metas)
        metas_array = np.array(new_rows, dtype=object)

        table = Table.from_numpy(domain, X=np.empty((len(new_rows), 0)), metas=metas_array)

        new_corpus: Corpus = Corpus.from_table(domain, table)
        new_corpus.attributes['language'] = self.corpus.attributes['language']
        new_corpus.set_text_features([new_corpus.columns.excerpt])

        self.results_display.setPlainText("\n---\n".join([f"{r[0]}\n(Similarity: {r[2]:.4f})" for r in results if r[2] >= self.threshold]))
        self.Outputs.data.send(new_corpus)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.text.corpus import Corpus

    corpus = Corpus('book-excerpts')
    WidgetPreview(OWReferenceLibrary).run(corpus)
