import unittest
from unittest import mock

from orangewidget.tests.base import WidgetTest
from orangecontrib.text import Corpus
from orangecontrib.text import preprocess as pp
from orangecontrib.nlp.widgets.owner import OWNERWidget, OllamaWorker, parse_entities


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


class TestParseEntities(unittest.TestCase):
    def test_object_with_entities_key(self) -> None:
        content = '{"entities": [{"text": "Japan", "label": "GPE"}, {"text": "BMW", "label": "ORG"}]}'
        self.assertEqual(parse_entities(content), "Japan (GPE), BMW (ORG)")

    def test_bare_array(self) -> None:
        self.assertEqual(parse_entities('[{"text": "Japan", "label": "GPE"}]'), "Japan (GPE)")

    def test_empty(self) -> None:
        self.assertEqual(parse_entities('{"entities": []}'), "")

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_entities("```json\n[]\n```")


class TestOllamaWorker(unittest.TestCase):
    def test_requests_json_format(self) -> None:
        response = mock.Mock()
        response.json.return_value = {"response": '{"entities": [{"text": "Japan", "label": "GPE"}]}'}
        worker = OllamaWorker(["Japan turns to beer alternatives"], model_name="llama3.1:8b")
        results: list[list[str]] = []
        worker.result.connect(results.append)
        with mock.patch("orangecontrib.nlp.widgets.owner.requests.post", return_value=response) as post:
            worker.run()
        self.assertEqual(post.call_args.kwargs["json"]["format"], "json")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "llama3.1:8b")
        self.assertEqual(results, [["Japan (GPE)"]])


if __name__ == "__main__":
    unittest.main()
