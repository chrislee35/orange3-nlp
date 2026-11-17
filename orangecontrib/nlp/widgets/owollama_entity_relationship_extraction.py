from orangecanvas.localization import Translator  # pylint: disable=wrong-import-order
_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator

import json

import requests
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QWidget
)
from Orange.data import Domain, StringVariable, Table
from Orange.widgets import settings, widget
from Orange.widgets.widget import Input, Output
from orangecontrib.text.corpus import Corpus

DEFAULT_PROMPT = (
    "Extract entity relationships from the text below as subject-verb-object triples.\n"
    "- 'subject' and 'object' are the entities (short noun phrases).\n"
    "- 'verb' is a short, lower-case relation (e.g. 'founded', 'located_in').\n"
    "- 'sentence' is the sentence from the text that supports the triple.\n"
    "Only extract relations that are stated in the text."
)

# JSON schema passed to Ollama's structured-output `format` option.
RELATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "verb": {"type": "string"},
                    "object": {"type": "string"},
                    "sentence": {"type": "string"},
                },
                "required": ["subject", "verb", "object"],
            },
        }
    },
    "required": ["relations"],
}


def parse_relations(content: str) -> list[tuple[str, str, str, str]]:
    """Parse a model response into (subject, verb, object, sentence) tuples.

    Accepts either {"relations": [...]} or a bare list, and skips malformed items.
    """
    data = json.loads(content)
    if isinstance(data, dict):
        data = data.get("relations", [])
    if not isinstance(data, list):
        raise ValueError("Response did not contain a list of relations")

    relations = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        verb = str(item.get("verb", "")).strip()
        obj = str(item.get("object", "")).strip()
        if subject and verb and obj:
            relations.append((subject, verb, obj, str(item.get("sentence", "")).strip()))
    return relations


class OllamaEntityRelationshipWorker(QThread):
    result_ready = pyqtSignal(list)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, texts, host, port, model, prompt):
        super().__init__()
        self.texts = texts
        self.host = host
        self.port = port
        self.model = model
        self.prompt = prompt
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def extract(self, text: str) -> list[tuple[str, str, str, str]]:
        response = requests.post(
            f"http://{self.host}:{self.port}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{self.prompt}\n\nText:\n{text}\n",
                "stream": False,
                "format": RELATIONS_SCHEMA,
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        response.raise_for_status()
        return parse_relations(response.json().get("response", ""))

    def run(self):
        results = []
        errors = []
        total = len(self.texts)
        for chunk_id, text in enumerate(self.texts):
            if self._cancel:
                return
            if text and text.strip():
                try:
                    for subj, verb, obj, sent in self.extract(text):
                        results.append((chunk_id, subj, verb, obj, sent))
                except Exception as e:  # keep going; report per-document failures
                    errors.append(f"Document {chunk_id}: {e}")
            self.progress.emit(int((chunk_id + 1) / total * 100))

        if errors:
            self.error.emit("\n".join(errors))
        self.result_ready.emit(results)


class OWOllamaEntityRelationshipExtraction(widget.OWWidget):
    name = _tr.m[87, "Ollama-based Entity-Relationship Extraction"]
    description = _tr.m[88, "Extract subject–verb–object relations using an Ollama model"]
    icon = "icons/nlp-relations.svg"
    priority = 51

    host = settings.Setting("localhost")
    port = settings.Setting(11434)
    model = settings.Setting("")
    prompt = settings.Setting(DEFAULT_PROMPT)

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        relations = Output("Relations", Table)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.worker = None
        self.layout_control_area()
        self.update_model_list()

    def layout_control_area(self):
        a = self.controlArea.layout().addWidget
        a(QLabel(_tr.m[26, "Host:"]))
        self.host_input = QLineEdit(self.host)
        self.host_input.editingFinished.connect(self.on_host_port_changed)
        a(self.host_input)
        a(QLabel(_tr.m[50, "Port:"]))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(int(self.port))
        self.port_input.editingFinished.connect(self.on_host_port_changed)
        a(self.port_input)
        a(QLabel(_tr.m[30, "Model:"]))
        self.model_combo = QComboBox()
        a(self.model_combo)
        a(QLabel(_tr.m[52, "Prompt:"]))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlainText(self.prompt)
        self.prompt_input.setMinimumHeight(140)
        a(self.prompt_input)

        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        self.run_button = QPushButton(_tr.m[89, "Extract"])
        self.run_button.clicked.connect(self.start_worker)
        row.addWidget(self.run_button)
        self.cancel_button = QPushButton(_tr.m[9, "Cancel"])
        self.cancel_button.clicked.connect(self.cancel_worker)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.cancel_button)
        a(buttons)

        self.info_label = QLabel(_tr.m[33, "No data on input yet."])
        a(self.info_label)
        a(QLabel(_tr.m[90, "Errors:"]))
        self.error_box = QTextEdit()
        self.error_box.setReadOnly(True)
        self.error_box.setStyleSheet("color: red;")
        self.error_box.setMaximumHeight(100)
        a(self.error_box)
        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def on_host_port_changed(self):
        self.host = self.host_input.text()
        self.port = self.port_input.value()
        self.update_model_list()

    def update_model_list(self):
        try:
            response = requests.get(f"http://{self.host}:{self.port}/api/tags", timeout=10)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
        except Exception as e:
            self.model_combo.clear()
            self.error_box.setPlainText(_tr.m[95, "Failed to fetch models from Ollama server: {error}"].format(error=e))
            return
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if self.model in models:
            self.model_combo.setCurrentText(self.model)
        elif models:
            self.model = models[0]

    @Inputs.corpus
    def set_corpus(self, corpus):
        self.cancel_worker()
        self.corpus = corpus
        self.error_box.clear()
        if corpus is None:
            self.info_label.setText(_tr.m[33, "No data on input yet."])
            self.Outputs.relations.send(None)
        else:
            self.info_label.setText(_tr.m[92, "Received {n} documents. Press Extract to run."].format(n=len(corpus)))

    def cancel_worker(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            self.progressBarFinished()
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.info_label.setText(_tr.m[51, "Processing cancelled."])

    def start_worker(self):
        if self.corpus is None:
            return
        self.cancel_worker()
        self.host = self.host_input.text()
        self.port = self.port_input.value()
        self.model = self.model_combo.currentText()
        self.prompt = self.prompt_input.toPlainText()
        if not self.model:
            self.error_box.setPlainText(_tr.m[94, "No Ollama model selected."])
            return

        self.error_box.clear()
        self.worker = OllamaEntityRelationshipWorker(
            list(self.corpus.documents), self.host, self.port, self.model, self.prompt
        )
        self.worker.progress.connect(self.progressBarSet)
        self.worker.error.connect(self.error_box.setPlainText)
        self.worker.result_ready.connect(self.on_result_ready)
        self.progressBarInit()
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.info_label.setText(_tr.m[91, "Extracting relations…"])
        self.worker.start()

    def on_result_ready(self, relations):
        self.progressBarFinished()
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.info_label.setText(_tr.m[93, "Extracted {n} relations."].format(n=len(relations)))

        # Same output schema as the rule-based extractor.
        domain = Domain(
            [],
            metas=[
                StringVariable("input_chunk_id"),
                StringVariable("subject"),
                StringVariable("verb"),
                StringVariable("object"),
                StringVariable("sentence"),
            ],
        )
        table = Table.from_list(domain, [
            [str(chunk_id), subj, verb, obj, sent]
            for chunk_id, subj, verb, obj, sent in relations
        ])
        self.Outputs.relations.send(table)

    def onDeleteWidget(self):
        self.cancel_worker()
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWOllamaEntityRelationshipExtraction).run(Corpus("book-excerpts"))
