import unittest
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text import Corpus
from orangecontrib.text.widgets.owpreprocess import OWPreprocess, TokenizerModule
from orangecontrib.nlp.widgets.owtoken_corpus import OWTokenCorpus
from unittest.mock import patch, PropertyMock, MagicMock, Mock

class TestOWTokenCorpusWidget(WidgetTest):
    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess."
           "storedsettings",
           PropertyMock(return_value={
               "preprocessors": [("preprocess.tokenize",
                                  {"method": TokenizerModule.Sentence})]
           }))
    def setUp(self):
        super().setUp()
        self.corpus = Corpus("andersen")
        self.preprocess_widget = self.create_widget(OWPreprocess)
        self.token_corpus_widget = self.create_widget(OWTokenCorpus)

    def test_sentence_tokenization_flow(self):
        # Simulate a tiny corpus
        corpus = self.corpus
        # Send corpus to OWPreprocess and enable sentence tokenization
        self.widget = self.preprocess_widget
        self.send_signal(self.preprocess_widget.Inputs.corpus, corpus)
        self.preprocess_widget.apply()  # Apply settings if needed

        # Get the output from OWPreprocess
        processed_corpus = self.get_output("Corpus", self.preprocess_widget)
        self.assertIsNotNone(processed_corpus)

        # Send processed corpus to OWTokenCorpus
        self.widget = self.token_corpus_widget
        self.send_signal(self.token_corpus_widget.Inputs.data, processed_corpus)

        # Get the output from OWTokenCorpus
        token_output = self.get_output("Tokenized Corpus", self.token_corpus_widget)
        self.assertIsNotNone(token_output)

        # Optional: Check that tokens are sentence-level
        for doc in token_output.documents:
            print(doc)

if __name__ == "__main__":
    unittest.main()