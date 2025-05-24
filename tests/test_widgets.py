import sys
import unittest
import random
import multiprocessing
import time
from orangewidget.tests.base import WidgetTest
from orangecontrib.text import Corpus
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

def make_sample(corpus_name, count):
    full_corpus = Corpus(corpus_name)
    indices = random.sample(range(len(full_corpus)), count)
    sample = full_corpus[indices]
    return sample

def widget_process_test(widget_cls_name, settings, queue):
    try:
        if widget_cls_name == 'OWAbstractiveSummary':
            from orangecontrib.nlp.widgets.owabstractive_summary import OWAbstractiveSummary
            widget_class = OWAbstractiveSummary
            input_data = make_sample('book-excerpts', 1)
        elif widget_cls_name == 'OWExtractiveSummary':
            from orangecontrib.nlp.widgets.owextractive_summary import OWExtractiveSummary
            widget_class = OWExtractiveSummary
            input_data = make_sample('book-excerpts', 1)
        elif widget_cls_name == 'OWNERWidget':
            from orangecontrib.nlp.widgets.owner import OWNERWidget
            widget_class = OWNERWidget
            input_data = make_sample('friends-transcripts', 10)
        elif widget_cls_name == 'OWOllamaRAG':
            from orangecontrib.nlp.widgets.owollama_rag import OWOllamaRAG
            widget_class = OWOllamaRAG
            input_data = make_sample('book-excerpts', 1)
        elif widget_cls_name == 'OWQuestionAnswer':
            from orangecontrib.nlp.widgets.owquestion_answer import OWQuestionAnswer
            widget_class = OWQuestionAnswer
            input_data = make_sample('andersen', 1)
        elif widget_cls_name == 'OWReferenceLibrary':
            from orangecontrib.nlp.widgets.owreference_library import OWReferenceLibrary
            widget_class = OWReferenceLibrary
            input_data = make_sample('book-excerpts', 1)
        elif widget_cls_name == 'OWPOSTagger':
            from orangecontrib.nlp.widgets.owpos_tagger import OWPOSTagger
            widget_class = OWPOSTagger
            input_data = make_sample('friends-transcripts', 10)

        output_count_expected = len(input_data)
        if widget_cls_name == 'OWOllamaRAG':
            output_count_expected = 1
        elif widget_cls_name == 'OWReferenceLibrary':
            output_count_expected = 5

        widget = widget_class()
        for k, v in settings.items():
            setattr(widget, k, v)
        widget.set_data(input_data)

        for name, output_signal in widget.Outputs.__dict__.items():
            if isinstance(output_signal, Output):
                print(name)
                output = widget.outputs.get(name)
                if not output_not_none(name, output):
                    queue.put((False, f"{widget.name} output {name} is None"))
                    print(f"{widget.name} output {name} is None")
                    return

        queue.put((True, f"{widget.name} passed"))
    except Exception as e:
        queue.put((False, f"{widget_cls.__name__} error: {str(e)}"))
        print(f"{widget_cls.__name__} error: {str(e)}")

def run_isolated(widget_cls_name, settings):
    time.sleep(10)

    queue = multiprocessing.Queue()

    widget_process_test(widget_cls_name, settings, queue)

    # p = multiprocessing.Process(
    #     target=widget_process_test, args=(widget_cls_name, settings, queue)
    # )
    # p.start()
    # p.join(600)  # timeout safeguard

    # if p.exitcode != 0:
    #     return False, f"{widget_cls.__name__} crashed with exit code {p.exitcode}"

    success, message = queue.get() if not queue.empty() else (False, "No result")
    return success, message

class TestNLPWidget(WidgetTest):
    def run_case(self, widget_cls_name, settings):
        success, message = run_isolated(widget_cls_name, settings)
        self.assertTrue(success, message)
    
    def test_create_tests(self):
        test_cases = [
            ['OWAbstractiveSummary', {'selected_framework': 'BART'}],
            ['OWAbstractiveSummary', {'selected_framework': 'T5'}],
            ['OWAbstractiveSummary', {'selected_framework': 'FLAN-T5'}],
            ['OWAbstractiveSummary', {'selected_framework': 'Pegasus'}],
            ['OWExtractiveSummary', {'selected_framework': 'Sumy'}],
            ['OWExtractiveSummary', {'selected_framework': 'Summa'}],
            ['OWExtractiveSummary', {'selected_framework': 'BART'}],
            ['OWNERWidget', {'selected_framework': 'NLTK'}],
            ['OWNERWidget', {'selected_framework': 'spaCy'}],
            ['OWNERWidget', {'selected_framework': 'Flair'}],
            ['OWPOSTagger', {'selected_framework': 'spaCy'}],
            ['OWPOSTagger', {'selected_framework': 'Stanza'}],
            ['OWPOSTagger', {'selected_framework': 'UDPipe'}],
            ['OWQuestionAnswer', {'question': "Who's the bad guy?"}],
            ['OWReferenceLibrary', {'selected_framework': 'sentence-transformers', 'query': 'farming'}],
            ['OWReferenceLibrary', {'selected_framework': 'e5-small-v2', 'query': 'farming'}],
            ['OWReferenceLibrary', {'selected_framework': 'nomic-embed-text', 'query': 'farming'}],
            ['OWReferenceLibrary', {'selected_framework': 'spacy', 'query': 'farming'}],
            # I do the Ollama models together since the server component will cache the model between processes, but only for a few minutes
            ['OWAbstractiveSummary', {'selected_framework': 'Ollama', 'selected_model': 'phi:latest'}],
            ['OWExtractiveSummary', {'selected_framework': 'Ollama', 'selected_model': 'phi:latest'}],
            ['OWNERWidget', {'selected_framework': 'Ollama', 'selected_model': 'phi:latest'}],
            ['OWOllamaRAG', {'model': 'phi:latest', 'prompt': 'who died?'}]
        ]
        for tc in test_cases:
            with self.subTest(tc=tc):
                self.run_case(*tc)

if __name__ == "__main__":
    unittest.main()