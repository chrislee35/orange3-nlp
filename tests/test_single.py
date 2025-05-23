import sys
import unittest
import random
from orangewidget.tests.base import WidgetTest
from orangecontrib.text import Corpus
from PyQt5.QtTest import QTest

widget_class, corpus_name, row_count, *args = sys.argv[1:]
args  = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
sys.argv = sys.argv[0:1]
row_count = int(row_count)

class TestNLPWidget(WidgetTest):
    def setUp(self):
        global widget_class
        super().setUp()

        full_corpus = Corpus(corpus_name)
        indices = random.sample(range(len(full_corpus)), row_count)
        sample = full_corpus[indices]

        if widget_class == 'OWAbstractiveSummary':
            from orangecontrib.nlp.widgets.owabstractive_summary import OWAbstractiveSummary
            widget_class = OWAbstractiveSummary
        elif widget_class == 'OWExtractiveSummary':
            from orangecontrib.nlp.widgets.owextractive_summary import OWExtractiveSummary
            widget_class = OWExtractiveSummary
        elif widget_class == 'OWNERWidget':
            from orangecontrib.nlp.widgets.owner import OWNERWidget
            widget_class = OWNERWidget
        elif widget_class == 'OWOllamaRAG':
            from orangecontrib.nlp.widgets.owollama_rag import OWOllamaRAG
            widget_class = OWOllamaRAG
        elif widget_class == 'OWQuestionAnswer':
            from orangecontrib.nlp.widgets.owquestion_answer import OWQuestionAnswer
            widget_class = OWQuestionAnswer
        elif widget_class == 'OWReferenceLibrary':
            from orangecontrib.nlp.widgets.owreference_library import OWReferenceLibrary
            widget_class = OWReferenceLibrary
        elif widget_class == 'OWPOSTagger':
            from orangecontrib.nlp.widgets.owpos_tagger import OWPOSTagger
            widget_class = OWPOSTagger
        self.widget = self.create_widget(widget_class)
        self.sample = sample

    def tearDown(self):
        self.widget.close()
        super().tearDown()

    def test_with_sample_and_settings(self):
        for k,v in args.items():
            if hasattr(self.widget, k):
                setattr(self.widget, k, v)
        self.send_signal("Corpus", self.sample)
        output = self.get_output(self.widget.Outputs.data)
        while output is None:
            QTest.qWait(3000)
            output = self.get_output(self.widget.Outputs.data, wait=3000)
        self.assertIsNotNone(output)
        self.assertEqual(len(self.sample), len(output))

if __name__ == "__main__":
    unittest.main()
