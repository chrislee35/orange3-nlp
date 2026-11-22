import unittest
from types import SimpleNamespace
from unittest import mock

from orangecontrib.nlp.widgets.owextractive_summary import OWExtractiveSummary
from orangecontrib.nlp.widgets.owabstractive_summary import OWAbstractiveSummary


def make_stub(cls: type, auto_apply: bool) -> SimpleNamespace:
    """The widget methods only need these attributes, so no Qt application is required."""
    stub = type("Stub", (SimpleNamespace,), {"set_data": cls.set_data})(
        corpus=None,
        set_partial_input_summary=mock.Mock(),
        set_input_summary=mock.Mock(),
        auto_apply=auto_apply,
        frameworks=["A", "B"],
        selected_framework="A",
        worker=None,
        infoLabel=mock.Mock(),
        ollama_panel=mock.Mock(),
        start_processing=mock.Mock(),
        cancel_processing=mock.Mock(),
        Outputs=SimpleNamespace(data=mock.Mock()),
    )
    stub.auto_apply_changed = lambda: cls.auto_apply_changed(stub)
    return stub


class TestSummaryApply(unittest.TestCase):
    def test_auto_apply_default_is_on(self) -> None:
        for cls in (OWExtractiveSummary, OWAbstractiveSummary):
            self.assertTrue(cls.auto_apply.default)

    def test_auto_apply_runs_on_new_data_and_framework_change(self) -> None:
        for cls in (OWExtractiveSummary, OWAbstractiveSummary):
            stub = make_stub(cls, True)
            stub.corpus = mock.MagicMock()
            cls.select_framework(stub, 1)
            self.assertEqual(stub.start_processing.call_count, 1)

    def test_manual_mode_waits_for_apply(self) -> None:
        for cls in (OWExtractiveSummary, OWAbstractiveSummary):
            stub = make_stub(cls, False)
            stub.corpus = mock.MagicMock()
            cls.select_framework(stub, 1)
            stub.start_processing.assert_not_called()
            self.assertIn("Apply", stub.infoLabel.setText.call_args.args[0])

    def test_switching_auto_apply_on_runs_pending_work(self) -> None:
        for cls in (OWExtractiveSummary, OWAbstractiveSummary):
            stub = make_stub(cls, True)
            stub.corpus = mock.MagicMock()
            cls.auto_apply_changed(stub)
            self.assertEqual(stub.start_processing.call_count, 1)
            stub = make_stub(cls, True)
            cls.auto_apply_changed(stub)  # no corpus: nothing to do
            stub.start_processing.assert_not_called()

    def test_new_data_in_manual_mode_waits_for_apply(self) -> None:
        for cls in (OWExtractiveSummary, OWAbstractiveSummary):
            corpus = mock.MagicMock()
            corpus.__len__.return_value = 3
            stub = make_stub(cls, False)
            stub.set_data(corpus)
            stub.start_processing.assert_not_called()
            self.assertIn("Received 3 documents", stub.infoLabel.setText.call_args.args[0])
            stub = make_stub(cls, True)
            stub.set_data(corpus)
            stub.start_processing.assert_called_once()


if __name__ == "__main__":
    unittest.main()
