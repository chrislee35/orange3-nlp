from AnyQt.QtWidgets import QLabel, QVBoxLayout, QRadioButton, QButtonGroup, QPushButton, QLineEdit, QCheckBox, QTextEdit, QTableWidget, QHeaderView, QTableWidgetItem
from AnyQt.QtCore import QThread, pyqtSignal, Qt
from Orange.widgets import widget, settings
from Orange.widgets.widget import Input, Output
from Orange.data import StringVariable
from orangecontrib.text.corpus import Corpus
import json

class QuestionAnswerWorker(QThread):
    progress = pyqtSignal(int)
    results = pyqtSignal(list)

    def __init__(self, texts, question, language="en"):
        super().__init__()
        self.texts = texts
        self.question = question
        self.language = language
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        from transformers import pipeline

        try:
            qa_pipeline = pipeline("question-answering")
        except Exception as e:
            self.results.emit([json.dumps({"error": f"Error loading QA pipeline: {e}"}) for _ in self.texts])
            return
        answers = []
        for text in self.texts:
            try:
                answer = qa_pipeline(question=self.question, context=text)
                answers.append(answer['answer'])
            except Exception as e:
                answers.append(f"Error during QA: {e}")
        self.results.emit(answers)

class OWQuestionAnswer(widget.OWWidget):
    name = "Question Answering"
    description = "Answer a question using a text corpus."
    icon = "icons/nlp-question-answer.svg"
    priority = 140

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        data = Output("Corpus", Corpus)

    want_main_area = True
    question = settings.Setting("")
    all_documents = settings.Setting(False)
    current_row = settings.Setting(0)

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.worker = None
        self.layout_control_area()
        self.layout_main_area()

    def layout_control_area(self):
        self.all_docs_checkbox = QCheckBox("All documents")
        self.all_docs_checkbox.setChecked(self.all_documents)
        self.all_docs_checkbox.stateChanged.connect(self.on_checkbox_change)
        self.controlArea.layout().addWidget(self.all_docs_checkbox)
        
        self.table = QTableWidget(self)
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Text"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.controlArea.layout().setContentsMargins(0, 0, 0, 0)
        self.controlArea.layout().setSpacing(0)
        self.controlArea.layout().addWidget(self.table)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)

    def layout_main_area(self):
        self.textbox = QLineEdit()
        self.textbox.setPlaceholderText("Enter your question here")
        self.textbox.setText(self.question)
        self.textbox.returnPressed.connect(self.ask_question)
        self.mainArea.layout().addWidget(self.textbox)

        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.ask_question)
        self.mainArea.layout().addWidget(self.ask_button)

        self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        self.mainArea.layout().addWidget(self.answer_box)

    def on_checkbox_change(self):
        self.all_documents = self.all_docs_checkbox.isChecked()
        if self.corpus and self.question:
            self.ask_question()

    def populate_table(self):
        self.table.clearContents()
        if not self.corpus or not len(self.corpus):
            self.table.setRowCount(0)
            return

        text_var = self.corpus.text_features[0]
        num_rows = min(30, len(self.corpus))
        self.table.setRowCount(num_rows)
        for i in range(num_rows):
            text = str(self.corpus[i][text_var])
            item = QTableWidgetItem(text)
            self.table.setItem(i, 0, item)
        self.table.selectRow(0)

    def on_selection_changed(self):
        selected = self.table.selectedItems()
        if selected:
            self.current_row = self.table.currentRow()
            self.ask_question()    

    @Inputs.data
    def set_data(self, data):
        self.corpus = data
        self.populate_table()
        if self.corpus and self.question:
            self.ask_question()

    def ask_question(self):
        self.question = self.textbox.text()
        if not self.question or not self.corpus:
            return

        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()

        language = getattr(self.corpus, "language", "en") or "en"
        texts = self.corpus.documents if self.all_documents else [self.corpus.documents[self.current_row]]

        self.progressBarInit()
        self.worker = QuestionAnswerWorker(texts, self.question, language)
        self.worker.progress.connect(self.progressBarSet)
        self.worker.results.connect(self.show_answer)
        self.worker.start()

    def show_answer(self, answers):
        self.answer_box.setPlainText("\n".join(answers[0:20]))
        self.progressBarFinished()

        # Add answer to corpus as a new column
        answer_var = StringVariable("Answer")
        if self.all_documents:
            new_data = self.corpus.add_column(answer_var, answers, to_metas=True)
        else:
            new_data = Corpus.from_table_rows(self.corpus, [self.current_row]).add_column(answer_var, answers, to_metas=True)
        self.Outputs.data.send(new_data)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    import random

    full_corpus = Corpus("book-excerpts")
    sample_corpus = full_corpus[random.sample(range(len(full_corpus)), 10)]

    WidgetPreview(OWQuestionAnswer).run(sample_corpus)
