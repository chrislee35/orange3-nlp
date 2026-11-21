import unittest
from unittest import mock

from orangewidget.tests.base import WidgetTest
from orangecontrib.text import Corpus
from orangecontrib.text import preprocess as pp
from orangecontrib.nlp.widgets.owner import OWNERWidget


class TestNERInput(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWNERWidget)

    def _texts_sent_to_worker(self, corpus: Corpus) -> list[str]:
        with mock.patch("orangecontrib.nlp.widgets.owner.SpacyWorker") as worker:
            self.widget.selected_framework = "spaCy"
            self.send_signal(self.widget.Inputs.data, corpus)
        return list(worker.call_args.args[0])

    def test_raw_corpus_uses_documents(self) -> None:
        corpus = Corpus.from_file("book-excerpts")[:3]
        self.assertEqual(self._texts_sent_to_worker(corpus), list(corpus.documents))

    def test_preprocessed_corpus_uses_transformed_text(self) -> None:
        corpus = Corpus.from_file("book-excerpts")[:3]
        processed = pp.PreprocessorList([pp.LowercaseTransformer(), pp.RegexpTokenizer()])(corpus)
        texts = self._texts_sent_to_worker(processed)
        self.assertEqual(texts, [d.lower() for d in corpus.documents])

    def test_no_transformation_keeps_case(self) -> None:
        corpus = Corpus.from_file("book-excerpts")[:3]
        processed = pp.PreprocessorList([pp.RegexpTokenizer(), pp.StopwordsFilter("en")])(corpus)
        self.assertEqual(self._texts_sent_to_worker(processed), list(corpus.documents))


if __name__ == "__main__":
    unittest.main()
