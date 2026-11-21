from typing import Any

from AnyQt.QtWidgets import QLabel, QRadioButton, QButtonGroup, QPushButton
from AnyQt.QtCore import QThread, pyqtSignal
from Orange.widgets import widget
from Orange.widgets.widget import Input, Output
from Orange.data import StringVariable
from orangecontrib.text.corpus import Corpus
import json
from orangecontrib.nlp.util import SpaCyDownloader, UDPipeDownloader

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import layout_of
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class POSWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, texts: list[str], language: str = "en") -> None:
        super().__init__()
        self.texts = texts
        self.language = language
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        raise NotImplementedError("Subclasses must implement this.")


class SpaCyPOSWorker(POSWorker):

    def run(self) -> None:
        import spacy

        model_name = SpaCyDownloader.require_model_name(self.language)
        try:
            nlp = SpaCyDownloader.load(model_name)
        except OSError as e:
            self.result.emit([json.dumps({"error": str(e)}) for _ in self.texts])
            return
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
                    "head": token.head.i - token.i,
                }
                for token in doc
            ]
            results.append(json.dumps(token_data))
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class StanzaPOSWorker(POSWorker):

    def run(self) -> None:
        import stanza

        stanza.download(self.language)
        nlp: Any = stanza.Pipeline(self.language)
        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            doc = nlp(text)
            token_data = []
            for sent in doc.sentences:
                for word in sent.words:
                    token_data.append(
                        {
                            "text": word.text,
                            "lemma": word.lemma,
                            "pos": word.upos,
                            "tag": word.xpos,
                            "dep": word.deprel,
                            "head": word.head - word.id if word.head != 0 else 0,
                        }
                    )
            results.append(json.dumps(token_data))
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class UDPipePOSWorker(POSWorker):

    def run(self) -> None:
        from ufal.udpipe import Model, Pipeline  # pyright: ignore[reportAttributeAccessIssue]
        import os

        model_path = UDPipeDownloader.model_path(self.language)
        if not os.path.exists(model_path):
            self.progress.emit(1)
            if UDPipeDownloader.download(self.language):
                self.progress.emit(10)
            else:
                err = f"Failed to download UDPipe model for language {self.language}"
                self.result.emit([json.dumps({"error": err}) for _ in self.texts])
                return
        model = Model.load(model_path)
        pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        results = []
        for i, text in enumerate(self.texts):
            if self._is_cancelled:
                return
            processed = pipeline.process(text)
            token_data = []
            for line in processed.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.split("\t")
                if len(cols) < 10:
                    continue
                token_data.append(
                    {
                        "text": cols[1],
                        "lemma": cols[2],
                        "pos": cols[3],
                        "tag": cols[4],
                        "dep": cols[7],
                        "head": int(cols[6]) - int(cols[0]) if int(cols[6]) != 0 else 0,
                    }
                )
            results.append(json.dumps(token_data))
            self.progress.emit(int((i + 1) / len(self.texts) * 100))
        self.result.emit(results)


class OWPOSTagger(widget.OWWidget):
    name = _tr.m[46, "POS Tagger"]
    description = _tr.m[71, "Tag parts of speech using different frameworks."]
    icon = "icons/nlp-pos-tagger.svg"
    priority = 130

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Tagged Corpus", Corpus)

    want_main_area = False
    selected_framework = setting("spaCy")

    def __init__(self) -> None:
        super().__init__()
        self.corpus = None
        self.frameworks = ["spaCy", "Stanza", "UDPipe"]
        self.framework_buttons = QButtonGroup(self)
        for i, fw in enumerate(self.frameworks):
            btn = QRadioButton(fw)
            if fw == self.selected_framework:
                btn.setChecked(True)
            layout_of(self.controlArea).addWidget(btn)
            self.framework_buttons.addButton(btn, i)
        self.framework_buttons.buttonClicked[int].connect(self.select_framework)
        self.cancel_button = QPushButton(_tr.m[9, "Cancel"], self)
        self.cancel_button.clicked.connect(self.cancel_processing)
        layout_of(self.controlArea).addWidget(self.cancel_button)
        self.infoLabel = QLabel(_tr.m[33, "No data on input yet."], self)
        layout_of(self).addWidget(self.infoLabel)
        self.worker = None

    def select_framework(self, index: int) -> None:
        if self.worker and self.worker.isRunning():
            self.cancel_processing()
        self.selected_framework = self.frameworks[index]
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
        texts = self.corpus.documents
        language = getattr(self.corpus, "language", "en") or "en"
        framework = self.selected_framework
        if framework == "spaCy":
            self.worker = SpaCyPOSWorker(texts, language)
        elif framework == "Stanza":
            self.worker = StanzaPOSWorker(texts, language)
        elif framework == "UDPipe":
            self.worker = UDPipePOSWorker(texts, language)
        else:
            self.infoLabel.setText(_tr.m[80, "Unknown framework selected."])
            return
        self.progressBarInit()
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.process_result)
        self.worker.start()

    def update_progress(self, value: int) -> None:
        self.progressBarSet(value)

    def process_result(self, pos_json_list: list[str]) -> None:
        if self.corpus is None:
            return
        meta_var = StringVariable("POS Tags")
        new_data = self.corpus.add_column(meta_var, pos_json_list, to_metas=True)
        self.Outputs.data.send(new_data)
        self.infoLabel.setText(_tr.m[48, "POS tagging complete."])
        self.progressBarFinished()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    import random

    full_corpus = Corpus("friends-transcripts")
    indices = random.sample(range(len(full_corpus)), 10)
    sample_corpus = full_corpus[indices]
    WidgetPreview(OWPOSTagger).run(sample_corpus)
