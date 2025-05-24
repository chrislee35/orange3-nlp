import sys
import unittest
import random
from orangewidget.tests.base import WidgetTest
from orangecontrib.text import Corpus
from PyQt5.QtTest import QTest

widget_name, corpus_name, row_count, *args = sys.argv[1:]
args  = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
sys.argv = sys.argv[0:1]
row_count = int(row_count)

def print_progress(value):
    print(f"Progress: {value}%")

class TestNLPWidget(WidgetTest):
    def setUp(self):
        global widget_name
        super().setUp()

        full_corpus = Corpus(corpus_name)
        indices = random.sample(range(len(full_corpus)), row_count)
        sample = full_corpus[indices]

        self.output_count_expected = len(sample)
        print(widget_name)

        if widget_name == 'OWAbstractiveSummary':
            from orangecontrib.nlp.widgets.owabstractive_summary import OWAbstractiveSummary
            widget_class = OWAbstractiveSummary
        elif widget_name == 'OWExtractiveSummary':
            from orangecontrib.nlp.widgets.owextractive_summary import OWExtractiveSummary
            widget_class = OWExtractiveSummary
        elif widget_name == 'OWNERWidget':
            from orangecontrib.nlp.widgets.owner import OWNERWidget
            widget_class = OWNERWidget
        elif widget_name == 'OWOllamaRAG':
            from orangecontrib.nlp.widgets.owollama_rag import OWOllamaRAG
            widget_class = OWOllamaRAG
            self.output_count_expected = 1
        elif widget_name == 'OWQuestionAnswer':
            from orangecontrib.nlp.widgets.owquestion_answer import OWQuestionAnswer
            widget_class = OWQuestionAnswer
        elif widget_name == 'OWReferenceLibrary':
            from orangecontrib.nlp.widgets.owreference_library import OWReferenceLibrary
            widget_class = OWReferenceLibrary
            self.output_count_expected = 5
        elif widget_name == 'OWPOSTagger':
            from orangecontrib.nlp.widgets.owpos_tagger import OWPOSTagger
            widget_class = OWPOSTagger
        self.widget = self.create_widget(widget_class)
        # Get input signal names
        self.input_names = [signal.name for signal in self.widget.Inputs.__dict__.values() if hasattr(signal, 'name')]
        # Get output signal names
        self.output_names = [signal.name for signal in self.widget.Outputs.__dict__.values() if hasattr(signal, 'name')]

        self.sample = sample

    def tearDown(self):
        self.widget.close()
        super().tearDown()

    def test_with_sample_and_settings(self):
        for k,v in args.items():
            if hasattr(self.widget, k):
                setattr(self.widget, k, v)
        if hasattr(self.widget, 'update_progress'):
            self.widget.update_progress = print_progress
        self.send_signal(self.widget.Inputs.data, self.sample)
        output = self.get_output(self.widget.Outputs.data)
        while output is None:
            QTest.qWait(3000)
            output = self.get_output(self.widget.Outputs.data, wait=3000)
        self.assertIsNotNone(output)
        self.assertEqual(self.output_count_expected, len(output))

if __name__ == "__main__":
    unittest.main()
