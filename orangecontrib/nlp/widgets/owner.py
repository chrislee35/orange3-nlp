from AnyQt.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QLineEdit,
    QComboBox,
    QWidget,
)
from AnyQt.QtCore import QThread, pyqtSignal, Qt
from Orange.widgets import widget
from Orange.widgets.widget import Input, Output
from Orange.data import StringVariable
from orangecontrib.text.corpus import Corpus
import json
import requests

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class NERWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self.texts = texts
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        pass


class NLTKWorker(NERWorker):

    def run(self) -> None:
        import nltk

        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("maxent_ne_chunker")
        nltk.download("words")
        from nltk import word_tokenize, pos_tag, ne_chunk

        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            try:
                tree = ne_chunk(pos_tag(word_tokenize(text)))
                entities = [" ".join((c[0] for c in chunk.leaves())) + f" ({chunk.label()})" for chunk in tree if hasattr(chunk, "label")]
                results.append(", ".join(entities))
            except Exception as e:
                results.append(f"Error: {e}")
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class SpacyWorker(NERWorker):

    def run(self) -> None:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            try:
                doc = nlp(text)
                entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
                results.append(", ".join(entities))
            except Exception as e:
                results.append(f"Error: {e}")
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class FlairWorker(NERWorker):

    def run(self) -> None:
        from flair.models import SequenceTagger
        from flair.data import Sentence

        tagger = SequenceTagger.load("ner")
        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            try:
                sentence = Sentence(text)
                tagger.predict(sentence)
                entities = [f"{entity.text} ({entity.tag})" for entity in sentence.get_spans("ner")]
                results.append(", ".join(entities))
            except Exception as e:
                results.append(f"Error: {e}")
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


OLLAMA_PROMPT = (
    "Extract the named entities (people, organizations, locations, dates, money amounts) from the text. "
    "Use the labels PERSON, ORG, GPE, DATE, MONEY, or OTHER. Only include entities that appear in the text. "
    "Return a JSON object with a single key 'entities': an array of objects with 'text' and 'label' keys.\n"
    "\nText: {text}\n"
)


def parse_entities(content: str) -> str:
    """Turn the JSON an Ollama model returned into 'text (LABEL), ...'.

    Accepts the requested object ``{"entities": [...]}`` or a bare array of ``{"text", "label"}`` objects.
    """
    data = json.loads(content)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return ", ".join(f"{e['text']} ({e['label']})" for e in data)


class OllamaWorker(NERWorker):

    def __init__(self, texts: list[str], host: str = "localhost", port: str | int = "11434", model_name: str = "llama3.1:8b") -> None:
        super().__init__(texts)
        self.host = host
        self.port = port
        self.model_name = model_name

    def run(self) -> None:
        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            if not text:
                results.append("")
            else:
                try:
                    response = requests.post(
                        f"http://{self.host}:{self.port}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": OLLAMA_PROMPT.format(text=text),
                            "stream": False,
                            "format": "json",
                            "options": {"temperature": 0},
                        },
                        headers={"Content-Type": "application/json"},
                        timeout=300,
                    )
                    results.append(parse_entities(response.json().get("response", "")))
                except Exception as e:
                    results.append(f"Error: {e}")
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class OWNERWidget(widget.OWWidget):
    name = _tr.m[32, "Named Entity Recognition"]
    description = _tr.m[81, "Uses selected NER framework to extract named entities."]
    icon = "icons/nlp-ner.svg"
    priority = 120

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Corpus with Entities", Corpus)

    want_main_area = False
    selected_framework = setting("spaCy")
    ollama_host = setting("localhost")
    ollama_port = setting("11434")
    selected_model = setting("")

    def __init__(self) -> None:
        super().__init__()
        self.corpus = None
        self.frameworks = ["NLTK", "spaCy", "Flair", "Ollama"]
        self.worker = None
        self.layout_control_area()

    def layout_control_area(self) -> None:
        a = layout_of(self.controlArea).addWidget
        self.framework_buttons = QButtonGroup(self)
        for i, fw in enumerate(self.frameworks):
            btn = QRadioButton(fw)
            if fw == self.selected_framework:
                btn.setChecked(True)
            a(btn)
            self.framework_buttons.addButton(btn, i)
        self.framework_buttons.buttonClicked[int].connect(self.select_framework)
        self.cancel_button = QPushButton(_tr.m[9, "Cancel"], self)
        self.cancel_button.clicked.connect(self.cancel_processing)
        a(self.cancel_button)
        self.infoLabel = QLabel(_tr.m[33, "No data on input yet."], self)
        a(self.infoLabel)
        layout_of(self.controlArea).setAlignment(self.infoLabel, Qt.AlignmentFlag.AlignLeft)
        self.layout_ollama_config()
        a(self.ollama_panel)
        self.ollama_panel.setVisible(self.selected_framework == "Ollama")
        layout_of(self.controlArea).setAlignment(Qt.AlignmentFlag.AlignTop)

    def layout_ollama_config(self) -> None:
        self.ollama_panel = QWidget()
        ollama_layout = QVBoxLayout()
        self.ollama_panel.setLayout(ollama_layout)
        self.host_input = QLineEdit(self.ollama_host)
        self.port_input = QLineEdit(self.ollama_port)
        self.model_selector = QComboBox()
        self.host_input.setPlaceholderText(_tr.m[38, "Ollama Host"])
        self.port_input.setPlaceholderText(_tr.m[41, "Ollama Port"])
        ollama_layout.addWidget(QLabel(_tr.m[39, "Ollama Host:"]))
        ollama_layout.addWidget(self.host_input)
        ollama_layout.addWidget(QLabel(_tr.m[42, "Ollama Port:"]))
        ollama_layout.addWidget(self.port_input)
        ollama_layout.addWidget(QLabel(_tr.m[40, "Ollama Model:"]))
        ollama_layout.addWidget(self.model_selector)
        self.host_input.editingFinished.connect(self.update_model_list)
        self.port_input.editingFinished.connect(self.update_model_list)
        self.update_model_list()

    def update_model_list(self) -> None:
        host, port = (self.host_input.text(), self.port_input.text())
        try:
            r = requests.get(f"http://{host}:{port}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                self.model_selector.clear()
                for m in models:
                    self.model_selector.addItem(m["name"])
                if self.selected_model:
                    index = self.model_selector.findText(self.selected_model)
                    if index >= 0:
                        self.model_selector.setCurrentIndex(index)
        except Exception as e:
            print("Failed to fetch models from Ollama server:", e)

    def select_framework(self, index: int) -> None:
        if self.worker and self.worker.isRunning():
            self.cancel_processing()
        self.selected_framework = self.frameworks[index]
        self.ollama_panel.setVisible(self.selected_framework == "Ollama")
        if self.corpus is not None:
            self.start_processing()

    def cancel_processing(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            self.infoLabel.setText(_tr.m[51, "Processing cancelled."])
            self.progressBarInit()

    @Inputs.data
    def set_data(self, data: Corpus | None) -> None:
        self.corpus = data
        if self.corpus is not None:
            self.infoLabel.setText(_tr.m[85, f"Received {len(self.corpus)} documents."])
            self.start_processing()
        else:
            self.infoLabel.setText(_tr.m[33, "No data on input yet."])
            self.Outputs.data.send(None)

    def start_processing(self) -> None:
        if self.corpus is None:
            return
        if self.worker and self.worker.isRunning():
            self.cancel_processing()
        text_var = next((var for var in self.corpus.text_features), None)
        if text_var is None:
            self.infoLabel.setText(_tr.m[36, "No text attribute found."])
            self.Outputs.data.send(None)
            return
        # The text after the transformation stages of Preprocess Text (lowercase, accent, HTML, URL removal);
        # equals the raw documents when the corpus was not preprocessed.
        texts = list(self.corpus.pp_documents)
        framework = self.selected_framework
        if framework == "NLTK":
            self.worker = NLTKWorker(texts)
        elif framework == "spaCy":
            self.worker = SpacyWorker(texts)
        elif framework == "Flair":
            self.worker = FlairWorker(texts)
        elif framework == "Ollama":
            self.ollama_host = self.host_input.text()
            self.ollama_port = self.port_input.text()
            self.selected_model = self.model_selector.currentText() or self.selected_model
            self.worker = OllamaWorker(texts, self.ollama_host, self.ollama_port, self.selected_model)
        else:
            self.infoLabel.setText(_tr.m[80, "Unknown framework selected."])
            return
        self.progressBarInit()
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.process_result)
        self.worker.start()
        self.save_ollama_config()

    def save_ollama_config(self) -> None:
        self.ollama_host = self.host_input.text()
        self.ollama_port = self.port_input.text()
        self.selected_model = self.model_selector.currentText()

    def update_progress(self, value: int) -> None:
        self.progressBarSet(value)

    def process_result(self, entity_list: list[str]) -> None:
        if self.corpus is None:
            return
        new_data = self.corpus.add_column(StringVariable("Named Entities"), entity_list, to_metas=True)
        self.Outputs.data.send(new_data)
        self.infoLabel.setText(_tr.m[31, "NER processing complete."])
        self.progressBarFinished()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    import random

    full_corpus = Corpus("election-tweets-2016")
    indices = random.sample(range(len(full_corpus)), 10)
    sample_corpus = full_corpus[indices]
    WidgetPreview(OWNERWidget).run(sample_corpus)
