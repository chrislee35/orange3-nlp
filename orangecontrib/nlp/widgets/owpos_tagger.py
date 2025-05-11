from AnyQt.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QRadioButton, QButtonGroup, QPushButton, QLineEdit, QComboBox, QWidget, QGridLayout
from AnyQt.QtCore import QThread, pyqtSignal, Qt
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import StringVariable
from orangecontrib.text.corpus import Corpus
import numpy as np
import json
from orangecontrib.nlp.util import SpaCyDownloader, UDPipeDownloader

class POSWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, texts, language="en"):
        super().__init__()
        self.texts = texts
        self.language = language
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        raise NotImplementedError("Subclasses must implement this.")

class SpaCyPOSWorker(POSWorker):
    def run(self):
        import spacy
        model_name = SpaCyDownloader.model_name(self.language)
        try:
            nlp = spacy.load(model_name)
        except OSError:
            if not SpaCyDownloader.download(model_name):
                self.result.emit([json.dumps({"error": f"spaCy model for '{self.language}' not found."}) for _ in self.texts])
                return
            nlp = spacy.load(model_name)

        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            doc = nlp(text)
            token_data = [
                {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                    "head": token.head.i - token.i
                }
                for token in doc
            ]
            results.append(json.dumps(token_data))
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class StanzaPOSWorker(POSWorker):
    def run(self):
        import stanza
        stanza.download(self.language)
        nlp = stanza.Pipeline(self.language)
        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            doc = nlp(text)
            token_data = []
            for sent in doc.sentences:
                for word in sent.words:
                    token_data.append({
                        "text": word.text,
                        "lemma": word.lemma,
                        "pos": word.upos,
                        "tag": word.xpos,
                        "dep": word.deprel,
                        "head": word.head - word.id if word.head != 0 else 0
                    })
            results.append(json.dumps(token_data))
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)

class UDPipePOSWorker(POSWorker):
    def run(self):
        from ufal.udpipe import Model, Pipeline
        import os

        model_path = f"{self.language}.udpipe"
        if not os.path.exists(model_path):
            self.progress.emit(1)
            # Attempt to download model from the standard UDPipe model repository
            if UDPipeDownloader.download(self.language):
                self.progress.emit(10)
            else:
                err = f"Failed to download UDPipe model for language {self.language}"
                self.result.emit([json.dumps({"error": err}) for _ in self.texts])
                return

        model = Model.load(model_path)
        pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            processed = pipeline.process(text)
            token_data = []
            for line in processed.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.split('\t')
                if len(cols) < 10:
                    continue
                token_data.append({
                    "text": cols[1],
                    "lemma": cols[2],
                    "pos": cols[3],
                    "tag": cols[4],
                    "dep": cols[7],
                    "head": int(cols[6]) - int(cols[0])
                })
            results.append(json.dumps(token_data))
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)

class OWPOSTagger(widget.OWWidget):
    name = "POS Tagger"
    description = "Tag parts of speech using different frameworks."
    icon = "icons/nlp-pos-tagger.svg"
    priority = 130

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Tagged Corpus", Corpus)

    want_main_area = False
    selected_framework = settings.Setting(0)

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.frameworks = ["spaCy", "Stanza", "UDPipe"]

        self.framework_buttons = QButtonGroup(self)
        for i, fw in enumerate(self.frameworks):
            btn = QRadioButton(fw)
            if i == self.selected_framework:
                btn.setChecked(True)
            self.controlArea.layout().addWidget(btn)
            self.framework_buttons.addButton(btn, i)
        self.framework_buttons.buttonClicked[int].connect(self.select_framework)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.controlArea.layout().addWidget(self.cancel_button)

        self.infoLabel = QLabel("No data on input yet.", self)
        self.layout().addWidget(self.infoLabel)

        self.progressBar = QProgressBar(self)
        self.progressBar.setValue(0)
        self.layout().addWidget(self.progressBar)

        self.worker = None

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
            self.progressBar.setValue(0)

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
        language = getattr(self.corpus, "language", "en") or "en"
        framework = self.frameworks[self.selected_framework]

        if framework == "spaCy":
            self.worker = SpaCyPOSWorker(texts, language)
        elif framework == "Stanza":
            self.worker = StanzaPOSWorker(texts, language)
        elif framework == "UDPipe":
            self.worker = UDPipePOSWorker(texts, language)
        else:
            self.infoLabel.setText("Unknown framework selected.")
            return

        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.process_result)
        self.worker.start()

    def update_progress(self, value):
        self.progressBar.setValue(value)

    def process_result(self, pos_json_list):
        meta_var = StringVariable("POS Tags")
        new_data = self.corpus.add_column(meta_var, pos_json_list, to_metas=True)
        self.Outputs.data.send(new_data)
        self.infoLabel.setText("POS tagging complete.")
        self.progressBar.setValue(100)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    import random

    full_corpus = Corpus("friends-transcripts")
    indices = random.sample(range(len(full_corpus)), 10)
    sample_corpus = full_corpus[indices]

    WidgetPreview(OWPOSTagger).run(sample_corpus)

