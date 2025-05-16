from AnyQt.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox,
    QLabel, QTextEdit, QProgressBar, QSpinBox, QDoubleSpinBox
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import Domain, StringVariable, Table
from orangecontrib.text.corpus import Corpus
import requests
import numpy as np
import json


class OllamaWorker(QThread):
    result = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, host, port, model, prompt, references):
        super().__init__()
        self.host = host
        self.port = port
        self.model = model
        self.prompt = prompt
        self.references = references
        self._cancel = False

    def run(self):
        try:
            headers = {"Content-Type": "application/json"}
            prompt = f"{self.prompt}\n\nReferences:\n{self.references}"
            data = json.dumps({"model": self.model, "prompt": prompt})
            url = f"http://{self.host}:{self.port}/api/generate"
            with requests.post(f"http://{self.host}:{self.port}/api/generate", headers=headers, data=data, stream=True, timeout=60) as response:
                if response.status_code == 200:
                    output = ""
                    for line in response.iter_lines():
                        if self._cancel:
                            break
                        if line:
                            msg = json.loads(line.decode("utf-8"))
                            if "response" in msg:
                                text = msg["response"]
                                self.result.emit(text)
                                output += text
                    if not self._cancel:
                        self.finished.emit(output)
                else:
                    self.error.emit(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            self.error.emit("Generation error: " + str(e))

    def cancel(self):
        self._cancel = True


class OWOllamaRAG(widget.OWWidget):
    name = "Ollama RAG"
    description = "Send RAG prompt to an Ollama model."
    icon = "icons/nlp-rag.svg"
    priority = 160

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Corpus", Corpus)

    host = settings.Setting("localhost")
    port = settings.Setting(11434)
    model = settings.Setting("")
    prompt = settings.Setting("")

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.worker = None

        self.layout_control_area()
        self.layout_main_area()
        self.update_model_list()

    def layout_control_area(self):
        self.controlArea.layout().addWidget(QLabel("Host:"))
        self.host_input = QLineEdit(self.host)
        self.host_input.editingFinished.connect(self.on_host_port_changed)
        self.controlArea.layout().addWidget(self.host_input)

        self.controlArea.layout().addWidget(QLabel("Port:"))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(self.port)
        self.port_input.editingFinished.connect(self.on_host_port_changed)
        self.controlArea.layout().addWidget(self.port_input)

        self.controlArea.layout().addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.controlArea.layout().addWidget(self.model_combo)

        self.error_box = QTextEdit()
        self.error_box.setReadOnly(True)
        self.error_box.setStyleSheet("color: red;")
        self.controlArea.layout().addWidget(QLabel("Generation Errors:"))
        self.controlArea.layout().addWidget(self.error_box)


        self.controlArea.layout().setAlignment(Qt.AlignTop)

    def layout_main_area(self):
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlainText(self.prompt)
        self.mainArea.layout().addWidget(QLabel("Prompt:"))
        self.mainArea.layout().addWidget(self.prompt_input)

        buttons_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_response)
        buttons_layout.addWidget(self.generate_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_worker)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        self.mainArea.layout().addLayout(buttons_layout)

        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.mainArea.layout().addWidget(QLabel("Output:"))
        self.mainArea.layout().addWidget(self.output_display)

    def on_host_port_changed(self):
        self.host = self.host_input.text()
        self.port = self.port_input.value()
        self.update_model_list()

    def on_model_changed(self, text):
        self.model = text

    def update_model_list(self):
        try:
            url = f"http://{self.host}:{self.port}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            tags = response.json().get("models", [])
            models = [tag["name"] for tag in tags]
            self.model_combo.clear()
            self.model_combo.addItems(models)
            if self.model in models:
                self.model_combo.setCurrentText(self.model)
            elif models:
                self.model = models[0]
        except Exception:
            self.model_combo.clear()
            self.model_combo.addItem("Error fetching models")

    @Inputs.data
    def set_data(self, data):
        self.corpus = data

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()

    def generate_response(self):
        if not self.corpus or not self.model:
            return

        self.stop_worker()
        self.error_box.setPlainText("")

        self.prompt = self.prompt_input.toPlainText()
        self.host = self.host_input.text()
        self.port = self.port_input.value()

        references = "\n\n".join(self.corpus.documents)

        self.worker = OllamaWorker(
            self.host, self.port, self.model, self.prompt, references
        )
        self.output_display.clear()
        self.stop_button.setEnabled(True)
        self.worker.result.connect(self.output_display.insertPlainText)
        self.worker.error.connect(self.error_box.setPlainText)
        self.worker.finished.connect(self.send_output)
        self.worker.start()     

    def send_output(self, result):
        self.stop_button.setEnabled(False)
        self.error_box.setPlainText("")

        domain = Domain([], metas=[
            StringVariable("prompt"),
            StringVariable("model"),
            StringVariable("references"),
            StringVariable("output")
        ])
        metas = [[
            self.prompt,
            self.model,
            "\n\n".join(self.corpus.documents),
            result
        ]]
        metas_array = np.array(metas, dtype=object)
        table = Table.from_numpy(domain, X=np.empty((1, 0)), metas=metas_array)
        new_corpus = Corpus.from_table(domain, table)
        self.Outputs.data.send(new_corpus)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWOllamaRAG).run(Corpus("farming.tab"))
