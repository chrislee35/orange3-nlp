import unittest

import spacy

from orangecontrib.nlp.widgets.owrule_based_entity_relationship_extraction import (
    RuleBasedEntityRelationshipExtractionWorker as Worker,
)


class TestExtractSVO(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        worker = Worker.__new__(Worker)
        return [r[1:4] for r in worker.extract_svo(self.nlp, 0, text)]

    def test_active(self) -> None:
        self.assertIn(("the dog", "chase", "the cat"), self.extract("The dog chased the cat."))

    def test_passive_agent_is_the_subject(self) -> None:
        """spaCy labels the "by" of a passive agent `agent`, not `prep`."""
        self.assertEqual(self.extract("The cat was chased by the dog."), [("the dog", "chase", "the cat")])


if __name__ == "__main__":
    unittest.main()
