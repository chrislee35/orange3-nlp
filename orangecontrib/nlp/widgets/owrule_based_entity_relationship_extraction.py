from typing import Any

import threading
import spacy
from PyQt5.QtCore import QThread, pyqtSignal
from orangecontrib.nlp.util.spacy_downloader import SpaCyDownloader
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, Domain, StringVariable
from orangecontrib.text.corpus import Corpus

from orangecanvas.localization import Translator

_tr = Translator("orangecontrib.nlp", "biolab.si", "Orange")
del Translator


class RuleBasedEntityRelationshipExtractionWorker(QThread):
    result_ready = pyqtSignal(list)
    progress = pyqtSignal(int)

    _thread_local = threading.local()

    def __init__(self, corpus: Corpus) -> None:
        super().__init__()
        self.corpus = corpus
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def _get_model(self, language: str) -> Any:
        if not hasattr(self._thread_local, "model"):
            self._thread_local.model = {}
        if language not in self._thread_local.model:
            model_name = SpaCyDownloader.require_model_name(language)
            try:
                self._thread_local.model[language] = spacy.load(model_name)
            except Exception:
                SpaCyDownloader.download(model_name)
                self._thread_local.model[language] = spacy.load(model_name)
        return self._thread_local.model[language]

    def extract_svo(self, nlp: Any, chunk_id: int, text: str) -> list[tuple[int, str, str, str, str]]:

        # Verbs that are usually discourse-level, not semantic relations
        VERB_BLACKLIST = {
            "show",
            "suggest",
            "indicate",
            "find",
            "report",
            "observe",
            "demonstrate",
            "describe",
            "note",
        }

        def normalize(text: str) -> str:
            return " ".join(text.lower().split())

        def expand_span(token: Any) -> str:
            return normalize(" ".join(t.text for t in token.subtree))

        doc = nlp(text)
        results = []

        for sent in doc.sents:
            # --- Handle normal and passive verbs ---
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                verb_lemma = token.lemma_.lower()
                if verb_lemma in VERB_BLACKLIST:
                    continue

                subjects = []
                objects = []

                # Active / passive subjects and objects
                for child in token.children:
                    if child.dep_ in ("nsubj",):
                        subjects.append(expand_span(child))
                    elif child.dep_ in ("nsubjpass",):
                        objects.append(expand_span(child))
                    elif child.dep_ in ("dobj", "attr", "dative"):
                        objects.append(expand_span(child))

                # Passive agent: "influenced by temperature"
                for prep in token.children:
                    if prep.dep_ == "prep" and prep.text.lower() == "by":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                subjects.append(expand_span(pobj))
                                verb_lemma = token.lemma_.lower()
                                if objects:
                                    for obj in objects:
                                        results.append(
                                            (
                                                chunk_id,
                                                subjects[-1],
                                                verb_lemma,
                                                obj,
                                                sent.text,
                                            )
                                        )

                # Prepositional verb relations (migrate_to, linked_to)
                for prep in token.children:
                    if prep.dep_ != "prep":
                        continue
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj":
                            rel = f"{verb_lemma}_{prep.text.lower()}"
                            obj = expand_span(pobj)
                            for subj in subjects:
                                results.append((chunk_id, subj, rel, obj, sent.text))

                # Direct SVO relations
                for subj in subjects:
                    for obj in objects:
                        results.append((chunk_id, subj, verb_lemma, obj, sent.text))

            # --- Handle copular / attribution relations ---
            for token in sent:
                if token.dep_ == "cop" and token.head.pos_ in ("NOUN", "ADJ"):
                    head = token.head
                    subjects = [expand_span(child) for child in head.children if child.dep_ == "nsubj"]
                    attributes = [expand_span(head)]

                    for subj in subjects:
                        for attr in attributes:
                            results.append((chunk_id, subj, "is", attr, sent.text))
        return results

    def run(self) -> None:
        nlp = self._get_model(language="en")
        results = []

        total = len(self.corpus)
        for chunk_id, text in enumerate(self.corpus.documents):
            if self._abort:
                return

            results.extend(self.extract_svo(nlp, chunk_id, text))

            self.progress.emit(int((chunk_id + 1) / total * 100))

        self.result_ready.emit(results)


class OWRuleBasedEntityRelationshipExtraction(OWWidget):
    name = _tr.m[96, "Rule-based Entity-Relationship Extraction"]
    description = _tr.m[97, "Extract subject–verb–object relations using spaCy"]
    icon = "icons/nlp-relations.svg"
    priority = 50

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        relations = Output("Relations", Table)

    def __init__(self) -> None:
        super().__init__()
        self.worker = None
        self.corpus = None

    @Inputs.corpus
    def set_corpus(self, corpus: Corpus | None) -> None:
        self.corpus = corpus
        if corpus is None:
            self.Outputs.relations.send(None)
            return

        self.start_worker()

    def start_worker(self) -> None:
        if self.corpus is None:
            return
        if self.worker:
            self.worker.abort()
            self.worker.wait()

        self.worker = RuleBasedEntityRelationshipExtractionWorker(self.corpus)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

    def on_progress(self, value: int) -> None:
        self.progressBarSet(value)

    def on_result_ready(self, relations: list[tuple[int, str, str, str, str]]) -> None:
        self.progressBarFinished()

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

        table = Table.from_list(
            domain,
            [[str(chunk_id), subj, verb, obj, sent] for chunk_id, subj, verb, obj, sent in relations],
        )

        self.Outputs.relations.send(table)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    corpus = Corpus("andersen")
    WidgetPreview(OWRuleBasedEntityRelationshipExtraction).run(corpus)
