from typing import Any

import json
import spacy
from spacy.tokens import Doc
from spacy import displacy
from AnyQt.QtWidgets import QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtWebEngineWidgets import QWebEngineView
from Orange.widgets import widget
from Orange.widgets.widget import Input
from orangecontrib.text.corpus import Corpus
from orangecontrib.nlp.util import SpaCyDownloader

from orangecanvas.localization import Translator
from orangecontrib.nlp.util.qt_helpers import box_layout_of, layout_of

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class OWPOSViewer(widget.OWWidget):
    name = _tr.m[47, "POS Viewer"]
    description = _tr.m[82, "Visualize part-of-speech tags and dependency parses."]
    icon = "icons/nlp-pos-viewer.svg"
    priority = 140

    class Inputs:
        data = Input("Tagged Corpus", Corpus)

    want_main_area = True
    resizing_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.corpus = None
        self.current_row = 0
        self.layout_control_area()
        self.layout_main_area()

    def layout_control_area(self) -> None:
        self.table = QTableWidget(self)
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Text"])
        vertical_header = self.table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        horizontal_header = self.table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(0, QHeaderView.Stretch)
        layout_of(self.controlArea).setContentsMargins(0, 0, 0, 0)
        layout_of(self.controlArea).setSpacing(0)
        layout_of(self.controlArea).addWidget(self.table)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)

    def layout_main_area(self) -> None:
        self.webview = QWebEngineView(self)
        self.webview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.webview.setAutoFillBackground(True)
        layout = box_layout_of(self.mainArea)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.webview, stretch=1)

    @Inputs.data
    def set_data(self, data: Corpus | None) -> None:
        self.corpus = data
        self.populate_table()
        self.update_display()

    def populate_table(self) -> None:
        self.table.clearContents()
        if not self.corpus or not len(self.corpus):
            self.table.setRowCount(0)
            return
        text_var = self.corpus.text_features[0]
        num_rows = min(30, len(self.corpus))
        self.table.setRowCount(num_rows)
        for i in range(num_rows):
            row: Any = self.corpus[i]
            text = str(row[text_var])
            item = QTableWidgetItem(text)
            self.table.setItem(i, 0, item)
        self.table.selectRow(0)

    def on_selection_changed(self) -> None:
        selected = self.table.selectedItems()
        if selected:
            self.current_row = self.table.currentRow()
            self.update_display()

    def update_display(self) -> None:
        if not self.corpus or not len(self.corpus):
            self.webview.setHtml(_tr.m[2, "<h3>No data available.</h3>"])
            return
        model_name = SpaCyDownloader.require_model_name(self.corpus.language)
        try:
            nlp = spacy.load(model_name)
        except OSError:
            if not SpaCyDownloader.download(model_name):
                self.webview.setHtml(_tr.m[84, f"<h3>spaCy {self.corpus.language} model not found.</h3>"])
                return
            nlp = spacy.load(model_name)
        pos_column = None
        for m in self.corpus.domain.metas:
            if m.name == "POS Tags":
                pos_column = m
                break
        if pos_column is None:
            self.webview.setHtml(_tr.m[1, "<h3>No POS Tags metadata found.</h3>"])
            return
        try:
            row: Any = self.corpus[self.current_row]
            cell = row[pos_column]
            json_data = json.loads(cell.value)
        except Exception as e:
            self.webview.setHtml(_tr.m[83, f"<h3>Invalid POS JSON format:<br>{e}</h3>"])
            return
        words = [t["text"] for t in json_data]
        heads = [i + t["head"] for i, t in enumerate(json_data)]
        deps = [t["dep"] for t in json_data]
        doc = Doc(nlp.vocab, words=words)
        for i, token in enumerate(doc):
            token.dep_ = deps[i]
            token.head = doc[heads[i]] if 0 <= heads[i] < len(doc) else token
        palette = self.palette()
        bg_color = palette.window().color().name()
        fg_color = palette.windowText().color().name()
        options = {
            "compact": True,
            "bg": bg_color,
            "color": fg_color,
            "font": "Source Sans Pro",
        }
        html = displacy.render(doc, style="dep", page=False, options=options)
        styled_html = f"\n<html>\n<head>\n  <style>\n    body {{\n      background-color: {bg_color};\n      color: {fg_color};\n      margin: 0;\n      padding: 10px;\n      font: Source Sans Pro;\n    }}\n    .displacy-word, .displacy-arrow {{\n      color: {fg_color};\n    }}\n  </style>\n</head>\n<body>\n  {html}\n</body>\n</html>\n"
        self.webview.setHtml(styled_html)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWPOSViewer).run()
