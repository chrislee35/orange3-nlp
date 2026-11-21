import unittest
from unittest import mock

from orangecontrib.nlp.widgets.owabstractive_summary import PegasusWorker


class TestPegasusWorker(unittest.TestCase):
    def test_truncates_by_tokens_and_avoids_repetition(self) -> None:
        summarizer = mock.Mock(return_value=[{"summary_text": "A summary."}])
        long_text = "This is a sentence about the economy. " * 80
        worker = PegasusWorker([long_text])
        results: list[list[str]] = []
        worker.result.connect(results.append)
        with mock.patch("transformers.pipeline", return_value=summarizer) as pipeline:
            worker.run()
        self.assertEqual(pipeline.call_args.args[:1], ("summarization",))
        self.assertEqual(pipeline.call_args.kwargs["model"], "google/pegasus-xsum")
        kwargs = summarizer.call_args.kwargs
        self.assertTrue(kwargs["truncation"])
        self.assertEqual(kwargs["no_repeat_ngram_size"], 3)
        self.assertLessEqual(len(summarizer.call_args.args[0]), 3000)
        self.assertEqual(results, [["A summary."]])

    def test_short_text_is_returned_unchanged(self) -> None:
        summarizer = mock.Mock()
        worker = PegasusWorker(["Too short."])
        results: list[list[str]] = []
        worker.result.connect(results.append)
        with mock.patch("transformers.pipeline", return_value=summarizer):
            worker.run()
        summarizer.assert_not_called()
        self.assertEqual(results, [["Too short."]])


if __name__ == "__main__":
    unittest.main()
