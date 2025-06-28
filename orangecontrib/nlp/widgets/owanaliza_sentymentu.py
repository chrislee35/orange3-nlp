import Orange.data
from Orange.widgets import widget
from Orange.widgets.widget import Input, Output
from orangecontrib.text import Corpus
from PyQt5.QtCore import QThread, pyqtSignal
from sentimentpl.models import SentimentPLModel
import numpy as np

class SentimentWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, texts):
        super().__init__()
        self.texts = texts
        self.model = SentimentPLModel(from_pretrained='latest')

    def run(self):
        results = []
        total = len(self.texts)
        for i, text in enumerate(self.texts):
            score = self.model(text).item()
            results.append(score)
            self.progress.emit(int((i + 1) / total * 100))
        self.result.emit(results)


class OWAnalizaSentymentu(widget.OWWidget):
    name = "Analiza Sentymentu"
    description = "Analiza sentymentu tekstu w języku polskim."
    icon = "icons/nlp-analiza-sentymentu.svg"
    priority = 10

    want_main_area = False
    want_control_area = False

    class Inputs:
        data = Input("Korpus", Corpus)

    class Outputs:
        data = Output("Korpus z sentymentem", Corpus)

    def __init__(self):
        super().__init__()
        self.data = None
        self.worker = None
        #self.infoBox = gui.widgetLabel(self.controlArea, "Oczekiwanie na dane...")

    @Inputs.data
    def set_data(self, data):
        self.data = data
        if data is not None:
            self.run_analysis()

    def run_analysis(self):
        self.worker = SentimentWorker(self.data.documents)
        self.worker.progress.connect(self.report_progress)
        self.worker.result.connect(self.handle_result)
        self.progressBarInit()
        self.worker.start()

    def report_progress(self, percent):
        self.progressBarSet(percent)

    def handle_result(self, scores):
        # Prepare new columns
        new_data = self.data.copy()

        values = np.array(scores)
        new_column = Orange.data.ContinuousVariable("score")
        new_data = new_data.add_column(new_column, values)
        
        self.Outputs.data.send(new_data)
        self.progressBarFinished()

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    import random

    full_corpus = Corpus("datasets/recenzja_produktu.tab")
    indices = random.sample(range(len(full_corpus)), 10)
    sample_corpus = full_corpus[indices]
    WidgetPreview(OWAnalizaSentymentu).run(sample_corpus)