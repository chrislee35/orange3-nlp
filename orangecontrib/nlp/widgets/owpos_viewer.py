from AnyQt.QtWidgets import QLabel, QVBoxLayout, QSizePolicy, QWidget, QTableWidget, QTableWidgetItem, QHeaderView
from AnyQt.QtCore import Qt
from Orange.widgets import widget
from Orange.widgets.widget import Input
from orangecontrib.text.corpus import Corpus
import json
import spacy
from spacy import displacy
from PyQt5.QtWebEngineWidgets import QWebEngineView
from orangecontrib.nlp.util import SpaCyDownloader


class OWPOSViewer(widget.OWWidget):
    name = "POS Viewer"
    description = "Visualize part-of-speech tags and dependency parses."
    icon = "icons/nlp-pos-visualizer.svg"
    priority = 140

    class Inputs:
        data = Input("Tagged Corpus", Corpus)

    want_main_area = True
    resizing_enabled = True

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.current_row = 0

        self.layout_control_area()
        self.layout_main_area()

    def layout_control_area(self):
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
        self.webview = QWebEngineView(self)
        self.webview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.webview.setAutoFillBackground(True)
        palette = self.palette()
        bg_color = palette.window().color().name()

        layout = self.mainArea.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.webview, stretch=1)

    @Inputs.data
    def set_data(self, data):
        self.corpus = data
        self.populate_table()
        self.update_display()

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
            self.update_display()

    def update_display(self):
        if not self.corpus or not len(self.corpus):
            self.webview.setHtml("<h3>No data available.</h3>")
            return

        # Load spaCy English model
        model_name = SpaCyDownloader.model_name(self.corpus.language)
        try:
            nlp = spacy.load(model_name)
        except OSError:
            if not SpaCyDownloader.download(model_name):
                self.webview.setHtml(f"<h3>spaCy {self.corpus.language} model not found.</h3>")
                return
            nlp = space.load(model_name)

        # Find the POS Tags column
        pos_column = None
        for m in self.corpus.domain.metas:
            if m.name == "POS Tags":
                pos_column = m
                break

        if pos_column is None:
            self.webview.setHtml("<h3>No POS Tags metadata found.</h3>")
            return

        # Just visualize the first document's parse
        try:
            json_data = json.loads(self.corpus[self.current_row][pos_column].value)
        except Exception as e:
            self.webview.setHtml(f"<h3>Invalid POS JSON format:<br>{e}</h3>")
            return

        # Convert back to Doc object for visualization
        words = [t["text"] for t in json_data]
        heads = [i + t["head"] for i, t in enumerate(json_data)]
        deps = [t["dep"] for t in json_data]
        doc = spacy.tokens.Doc(nlp.vocab, words=words)

        # Inject heads and dependencies manually
        for i, token in enumerate(doc):
            token.dep_ = deps[i]
            token.head = doc[heads[i]] if 0 <= heads[i] < len(doc) else token

        
        palette = self.palette()
        bg_color = palette.window().color().name()
        fg_color = palette.windowText().color().name()

        

        options = {"compact": True, "bg": bg_color, "color": fg_color, "font": "Source Sans Pro"}
        html = displacy.render(doc, style="dep", page=False, options=options)
        styled_html = f'''
<html>
<head>
  <style>
    body {{
      background-color: {bg_color};
      color: {fg_color};
      margin: 0;
      padding: 10px;
      font: Source Sans Pro;
    }}
    .displacy-word, .displacy-arrow {{
      color: {fg_color};
    }}
  </style>
</head>
<body>
  {html}
</body>
</html>
'''
        self.webview.setHtml(styled_html)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPOSViewer).run()