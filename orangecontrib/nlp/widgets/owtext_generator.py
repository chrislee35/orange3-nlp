import requests
from AnyQt.QtCore import Qt, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from Orange.widgets import widget
from Orange.widgets.widget import Input, Msg, Output
from orangecanvas.localization import Translator
from orangecontrib.text.corpus import Corpus

from orangecontrib.nlp.util.qt_helpers import box_layout_of, layout_of
from orangecontrib.nlp.util.text_generation import (
    BACKENDS,
    DEFAULT_HF_MODEL,
    HF_BACKEND,
    HF_MODELS,
    OLLAMA_BACKEND,
    GenerationSettings,
    Generator,
    Job,
    build_output,
    make_generator,
    make_jobs,
    render_prompt,
    row_values,
)
from orangecontrib.nlp.util.typed_settings import setting

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class GenerationWorker(QThread):
    token = pyqtSignal(int, str)
    progress = pyqtSignal(int)
    done = pyqtSignal(list)

    def __init__(self, generator: Generator, jobs: list[Job], settings: GenerationSettings) -> None:
        super().__init__()
        self.generator = generator
        self.jobs = jobs
        self.settings = settings
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        texts: list[str] = []
        for index, job in enumerate(self.jobs):
            if self._cancelled:
                return
            try:
                text = self.generator.generate(
                    job.prompt,
                    self.settings,
                    job.sample,
                    lambda piece, index=index: self.token.emit(index, piece),
                    self.is_cancelled,
                )
            except Exception as e:
                text = f"Error: {e}"
                self.token.emit(index, text)
            texts.append(text)
            self.progress.emit(int((index + 1) / len(self.jobs) * 100))
        if not self._cancelled:
            self.done.emit(texts)


class OWTextGenerator(widget.OWWidget):
    name = _tr.m[98, "Text Generator"]
    description = _tr.m[99, "Generate text from a prompt with a local language model."]
    icon = "icons/nlp-generate.svg"
    priority = 165

    backend = setting(OLLAMA_BACKEND)
    ollama_host = setting("localhost")
    ollama_port = setting(11434)
    ollama_model = setting("")
    hf_model = setting(DEFAULT_HF_MODEL)
    prompt = setting("Once upon a time")
    temperature = setting(0.7)
    max_new_tokens = setting(100)
    top_p = setting(0.9)
    use_seed = setting(False)
    seed = setting(42)
    samples = setting(1)

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Corpus", Corpus)

    class Warning(widget.OWWidget.Warning):
        unknown_placeholder = Msg(_tr.m[113, "Unknown placeholder(s): {}"])
        no_prompt = Msg(_tr.m[114, "Enter a prompt."])
        no_model = Msg(_tr.m[115, "Choose a model."])

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()
        self.corpus: Corpus | None = None
        self.worker: GenerationWorker | None = None
        self.jobs: list[Job] = []
        self.prompts: list[str] = []
        self._shown_job = -1
        self.layout_control_area()
        self.layout_main_area()
        self.update_info()
        self.update_ollama_models()
        self.update_backend_panels()

    def layout_control_area(self) -> None:
        add = layout_of(self.controlArea).addWidget
        add(QLabel(_tr.m[100, "Backend:"]))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(BACKENDS)
        self.backend_combo.setCurrentText(self.backend)
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        add(self.backend_combo)

        self.ollama_panel = QWidget()
        ollama_layout = box_layout_of_new(self.ollama_panel)
        ollama_layout.addWidget(QLabel(_tr.m[39, "Ollama Host:"]))
        self.host_input = QLineEdit(self.ollama_host)
        self.host_input.editingFinished.connect(self.on_ollama_server_changed)
        ollama_layout.addWidget(self.host_input)
        ollama_layout.addWidget(QLabel(_tr.m[42, "Ollama Port:"]))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(int(self.ollama_port))
        self.port_input.editingFinished.connect(self.on_ollama_server_changed)
        ollama_layout.addWidget(self.port_input)
        ollama_layout.addWidget(QLabel(_tr.m[40, "Ollama Model:"]))
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.setEditable(True)
        self.ollama_model_combo.setCurrentText(self.ollama_model)
        self.ollama_model_combo.currentTextChanged.connect(self.on_ollama_model_changed)
        ollama_layout.addWidget(self.ollama_model_combo)
        add(self.ollama_panel)

        self.hf_panel = QWidget()
        hf_layout = box_layout_of_new(self.hf_panel)
        hf_layout.addWidget(QLabel(_tr.m[101, "Hugging Face model:"]))
        self.hf_model_combo = QComboBox()
        self.hf_model_combo.setEditable(True)
        self.hf_model_combo.addItems(HF_MODELS)
        self.hf_model_combo.setCurrentText(self.hf_model)
        self.hf_model_combo.currentTextChanged.connect(self.on_hf_model_changed)
        hf_layout.addWidget(self.hf_model_combo)
        add(self.hf_panel)

        add(QLabel(_tr.m[102, "Temperature:"]))
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setValue(float(self.temperature))
        self.temperature_spin.valueChanged.connect(self.on_temperature_changed)
        add(self.temperature_spin)

        add(QLabel(_tr.m[103, "Max new tokens:"]))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 4096)
        self.max_tokens_spin.setValue(int(self.max_new_tokens))
        self.max_tokens_spin.valueChanged.connect(self.on_max_tokens_changed)
        add(self.max_tokens_spin)

        add(QLabel(_tr.m[104, "Top-p:"]))
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.05, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.setValue(float(self.top_p))
        self.top_p_spin.valueChanged.connect(self.on_top_p_changed)
        add(self.top_p_spin)

        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        self.seed_check = QCheckBox(_tr.m[105, "Fixed seed"])
        self.seed_check.setChecked(bool(self.use_seed))
        self.seed_check.toggled.connect(self.on_seed_toggled)
        seed_layout.addWidget(self.seed_check)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(int(self.seed))
        self.seed_spin.setEnabled(bool(self.use_seed))
        self.seed_spin.valueChanged.connect(self.on_seed_changed)
        seed_layout.addWidget(self.seed_spin)
        add(seed_row)

        add(QLabel(_tr.m[106, "Samples per prompt:"]))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 50)
        self.samples_spin.setValue(int(self.samples))
        self.samples_spin.valueChanged.connect(self.on_samples_changed)
        add(self.samples_spin)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        add(self.info_label)
        layout_of(self.controlArea).setAlignment(Qt.AlignmentFlag.AlignTop)

    def layout_main_area(self) -> None:
        add = layout_of(self.mainArea).addWidget
        add(QLabel(_tr.m[107, "Prompt template:"]))
        self.prompt_input = QTextEdit()
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.setPlainText(self.prompt)
        self.prompt_input.textChanged.connect(self.on_prompt_changed)
        add(self.prompt_input)
        buttons = QHBoxLayout()
        self.generate_button = QPushButton(_tr.m[108, "Generate"])
        self.generate_button.clicked.connect(self.generate)
        buttons.addWidget(self.generate_button)
        self.stop_button = QPushButton(_tr.m[109, "Stop"])
        self.stop_button.clicked.connect(self.stop_worker)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.stop_button)
        box_layout_of(self.mainArea).addLayout(buttons)
        add(QLabel(_tr.m[110, "Output:"]))
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        add(self.output_display)

    # --- settings callbacks -------------------------------------------------

    def on_backend_changed(self, text: str) -> None:
        self.backend = text
        self.update_backend_panels()

    def update_backend_panels(self) -> None:
        self.ollama_panel.setVisible(self.backend == OLLAMA_BACKEND)
        self.hf_panel.setVisible(self.backend == HF_BACKEND)

    def on_ollama_server_changed(self) -> None:
        self.ollama_host = self.host_input.text()
        self.ollama_port = self.port_input.value()
        self.update_ollama_models()

    def on_ollama_model_changed(self, text: str) -> None:
        self.ollama_model = text

    def on_hf_model_changed(self, text: str) -> None:
        self.hf_model = text

    def on_temperature_changed(self, value: float) -> None:
        self.temperature = value

    def on_max_tokens_changed(self, value: int) -> None:
        self.max_new_tokens = value

    def on_top_p_changed(self, value: float) -> None:
        self.top_p = value

    def on_seed_toggled(self, checked: bool) -> None:
        self.use_seed = checked
        self.seed_spin.setEnabled(checked)

    def on_seed_changed(self, value: int) -> None:
        self.seed = value

    def on_samples_changed(self, value: int) -> None:
        self.samples = value

    def on_prompt_changed(self) -> None:
        self.prompt = self.prompt_input.toPlainText()

    def update_ollama_models(self) -> None:
        try:
            response = requests.get(f"http://{self.ollama_host}:{self.ollama_port}/api/tags", timeout=5)
            response.raise_for_status()
            models = [tag["name"] for tag in response.json().get("models", [])]
        except Exception:
            return  # keep the editable combo: the user can still type a model name
        current = self.ollama_model
        self.ollama_model_combo.blockSignals(True)
        self.ollama_model_combo.clear()
        self.ollama_model_combo.addItems(models)
        if current in models or (current and not models):
            self.ollama_model_combo.setCurrentText(current)
        elif models:
            self.ollama_model = models[0]
            self.ollama_model_combo.setCurrentText(models[0])
        self.ollama_model_combo.blockSignals(False)

    # --- input ---------------------------------------------------------------

    @Inputs.data
    def set_data(self, data: Corpus | None) -> None:
        self.corpus = data
        self.update_info()

    def update_info(self) -> None:
        if self.corpus is None:
            self.info_label.setText(_tr.m[111, "No input: the prompt runs once."])
        else:
            self.info_label.setText(_tr.m[112, "Received {n} documents. Press Generate to run."].format(n=len(self.corpus)))

    # --- running -------------------------------------------------------------

    def current_model(self) -> str:
        return self.ollama_model_combo.currentText() if self.backend == OLLAMA_BACKEND else self.hf_model_combo.currentText()

    def generation_settings(self) -> GenerationSettings:
        return GenerationSettings(
            temperature=float(self.temperature),
            max_new_tokens=int(self.max_new_tokens),
            top_p=float(self.top_p),
            seed=int(self.seed) if self.use_seed else None,
        )

    def build_prompts(self) -> list[str]:
        """Render the template once per input row (or once, with no input). Sets the unknown-placeholder warning."""
        unknown: list[str] = []
        if self.corpus is None:
            prompt, unknown = render_prompt(self.prompt, {})
            prompts = [prompt]
        else:
            prompts = []
            for row in range(len(self.corpus)):
                prompt, names = render_prompt(self.prompt, row_values(self.corpus, row))
                prompts.append(prompt)
                unknown += [n for n in names if n not in unknown]
        if unknown:
            self.Warning.unknown_placeholder(", ".join("{" + n + "}" for n in unknown))
        return prompts

    def generate(self) -> None:
        self.Warning.clear()
        if not self.prompt.strip():
            self.Warning.no_prompt()
            return
        model = self.current_model().strip()
        if not model:
            self.Warning.no_model()
            return
        self.stop_worker()
        self.prompts = self.build_prompts()
        if self.corpus is not None and len(self.corpus) == 0:
            self.Outputs.data.send(None)
            return
        self.jobs = make_jobs(self.prompts, int(self.samples))
        generator = make_generator(self.backend, model, self.ollama_host, self.ollama_port)
        self.worker = GenerationWorker(generator, self.jobs, self.generation_settings())
        self.output_display.clear()
        self._shown_job = -1
        self.progressBarInit()
        self.generate_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.token.connect(self.show_token)
        self.worker.progress.connect(self.progressBarSet)
        self.worker.done.connect(self.send_output)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def show_token(self, index: int, piece: str) -> None:
        if index != self._shown_job:
            self._shown_job = index
            if len(self.jobs) > 1:
                job = self.jobs[index]
                label = f"Document {job.row + 1}, sample {job.sample + 1}" if self.corpus is not None else f"Sample {job.sample + 1}"
                self.output_display.append(f"\n--- {label} ---\n")
        self.output_display.insertPlainText(piece)

    def stop_worker(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()

    def on_worker_finished(self) -> None:
        self.progressBarFinished()
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def send_output(self, texts: list[str]) -> None:
        self.Outputs.data.send(build_output(self.corpus, self.jobs, texts, self.prompts))
        self.info_label.setText(_tr.m[116, "Generated {n} texts."].format(n=len(texts)))

    def onDeleteWidget(self) -> None:
        self.stop_worker()
        super().onDeleteWidget()


def box_layout_of_new(parent: QWidget) -> QVBoxLayout:
    """Give ``parent`` a zero-margin vertical box layout and return it."""
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    return layout


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWTextGenerator).run(Corpus("book-excerpts"))
