from AnyQt.QtWidgets import QLabel, QVBoxLayout, QGridLayout, QRadioButton, QButtonGroup, QPushButton, QLineEdit, QComboBox, QWidget
from AnyQt.QtCore import QThread, pyqtSignal, Qt
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import StringVariable
from orangecontrib.text.corpus import Corpus
import numpy as np
import json
import requests

class NERWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, texts):
        super().__init__()
        self.texts = texts
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        pass  # Implemented in subclass


class NLTKWorker(NERWorker):
    def run(self):
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        from nltk import word_tokenize, pos_tag, ne_chunk

        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            try:
                tree = ne_chunk(pos_tag(word_tokenize(text)))
                entities = [" ".join(c[0] for c in chunk.leaves()) + f" ({chunk.label()})"
                            for chunk in tree if hasattr(chunk, 'label')]
                results.append(", ".join(entities))
            except Exception as e:
                results.append(f"Error: {e}")
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class SpacyWorker(NERWorker):
    def run(self):
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
    def run(self):
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

class OllamaWorker(NERWorker):
    def __init__(self, texts, model_name="phi"):
        super().__init__(texts)
        self.model_name = model_name

    def run(self):
        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            if not text:
                results.append("")
            else:
                prompt = (
                    "Extract named entities from the following text. "
                    "Return a JSON array of objects with 'text' and 'label' keys.\n\n"
                    f"Text: {text}\n"
                )
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False
                        },
                        headers={"Content-Type": "application/json"}
                    )
                    content = response.json().get("response", "")
                    entities = json.loads(content)
                    formatted = ", ".join(f"{e['text']} ({e['label']})" for e in entities)
                    results.append(formatted)
                except Exception as e:
                    results.append(f"Error: {e}")
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class OWNERWidget(widget.OWWidget):
    name = "Named Entity Recognition"
    description = "Uses selected NER framework to extract named entities."
    icon = "icons/nlp-ner.svg"
    priority = 120

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Corpus with Entities", Corpus)

    want_main_area = False

    selected_framework = settings.Setting(0)
    ollama_host = settings.Setting("localhost")
    ollama_port = settings.Setting("11434")
    selected_model = settings.Setting("")

    def __init__(self):
        super().__init__()
        self.corpus = None

        self.frameworks = ["NLTK", "spaCy", "Flair", "Ollama"]
        
        self.control_widget = QWidget()
        layout = QGridLayout()
        layout.setVerticalSpacing(2)

        self.framework_buttons = QButtonGroup(self)
        for i, fw in enumerate(self.frameworks):
            btn = QRadioButton(fw)
            if i == self.selected_framework:
                btn.setChecked(True)
            layout.addWidget(btn, i, 1)
            self.framework_buttons.addButton(btn, i)
        self.framework_buttons.buttonClicked[int].connect(self.select_framework)

        i = len(self.frameworks)
        layout.addWidget(QLabel("Ollama Host:"), i, 0)
        self.host_input = QLineEdit(self.ollama_host)
        layout.addWidget(self.host_input, i, 1)
        self.host_input.editingFinished.connect(self.update_model_list)

        layout.addWidget(QLabel("Ollama Port:"), i+1, 0)
        self.port_input = QLineEdit(self.ollama_port)
        layout.addWidget(self.port_input, i+1, 1)
        self.port_input.editingFinished.connect(self.update_model_list)

        self.model_selector = QComboBox()
        layout.addWidget(QLabel("Active Model:"), i+2, 0)
        layout.addWidget(self.model_selector, i+2, 1)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_processing)
        layout.addWidget(self.cancel_button, i+3, 0, 1, 2)
        self.infoLabel = QLabel("No data on input yet.", self)
        layout.addWidget(self.infoLabel, i+4, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)

        
        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.addLayout(layout)

        self.control_widget.setLayout(control_layout)
        self.controlArea.layout().addWidget(self.control_widget)

        self.update_model_list()
        self.worker = None

    def update_model_list(self):
        host, port = self.host_input.text(), self.port_input.text()
        try:
            r = requests.get(f"http://{host}:{port}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                self.model_selector.clear()
                for m in models:
                    self.model_selector.addItem(m['name'])
                if self.selected_model:
                    index = self.model_selector.findText(self.selected_model)
                    if index >= 0:
                        self.model_selector.setCurrentIndex(index)
        except Exception as e:
            print("Failed to fetch models from Ollama server:", e)

    def select_framework(self, index):
        if self.worker and self.worker.isRunning():
            self.cancel_processing()
        self.selected_framework = index
        if self.corpus is not None:
            self.start_processing()

    def cancel_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            self.infoLabel.setText("Processing cancelled.")
            self.progressBarInit() # external progress bar

    @Inputs.data
    def set_data(self, data):
        self.corpus = data
        if self.corpus is not None:
            self.infoLabel.setText(f"Received {len(self.corpus)} documents.")
            self.start_processing()
        else:
            self.infoLabel.setText("No data on input yet.")
            self.Outputs.data.send(None)

    def start_processing(self):
        if self.worker and self.worker.isRunning():
            self.cancel_processing()

        text_var = next((var for var in self.corpus.text_features), None)
        if text_var is None:
            self.infoLabel.setText("No text attribute found.")
            self.Outputs.data.send(None)
            return
        texts = self.corpus.documents
        framework = self.frameworks[self.selected_framework]

        if framework == "NLTK":
            self.worker = NLTKWorker(texts)
        elif framework == "spaCy":
            self.worker = SpacyWorker(texts)
        elif framework == "Flair":
            self.worker = FlairWorker(texts)
        elif framework == "Ollama":
            self.worker = OllamaWorker(texts)
        else:
            self.infoLabel.setText("Unknown framework selected.")
            return
            
        self.progressBarInit()
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.process_result)
        self.worker.start()

    def update_progress(self, value):
        self.progressBarSet(value)

    def process_result(self, entity_list):
        new_data = self.corpus.add_column(StringVariable("Named Entities"), entity_list, to_metas=True)
        self.Outputs.data.send(new_data)
        self.infoLabel.setText("NER processing complete.")
        self.progressBarFinished()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    import random

    full_corpus = Corpus("election-tweets-2016")
    indices = random.sample(range(len(full_corpus)), 10)
    sample_corpus = full_corpus[indices]
    WidgetPreview(OWNERWidget).run(sample_corpus)
