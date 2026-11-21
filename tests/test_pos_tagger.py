import json
import sys
import unittest
from unittest import mock

from orangecontrib.nlp.widgets.owpos_tagger import UDPipePOSWorker

CONLLU = (
    "# text = The dog chased the cat.\n"
    "1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_\n"
    "2\tdog\tdog\tNOUN\tNN\t_\t3\tnsubj\t_\t_\n"
    "3\tchased\tchase\tVERB\tVBD\t_\t0\troot\t_\t_\n"
    "4\tthe\tthe\tDET\tDT\t_\t5\tdet\t_\t_\n"
    "5\tcat\tcat\tNOUN\tNN\t_\t3\tobj\t_\t_\n"
    "6\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_\n"
)


class TestUDPipeWorker(unittest.TestCase):
    def test_root_head_is_zero(self) -> None:
        """The root must point at itself (head 0), as with the spaCy and Stanza workers."""
        pipeline = mock.Mock()
        pipeline.process.return_value = CONLLU
        worker = UDPipePOSWorker(["The dog chased the cat."])
        results: list[list[str]] = []
        worker.result.connect(results.append)
        fake = mock.MagicMock()
        fake.Pipeline.return_value = pipeline
        with (
            mock.patch("os.path.exists", return_value=True),
            mock.patch.dict(sys.modules, {"ufal": mock.MagicMock(udpipe=fake), "ufal.udpipe": fake}),
        ):
            worker.run()
        tokens = json.loads(results[0][0])
        self.assertEqual([t["head"] for t in tokens], [1, 1, 0, 1, -2, -3])
        for i, t in enumerate(tokens):
            if t["dep"] == "root":
                self.assertEqual(i + t["head"], i)


if __name__ == "__main__":
    unittest.main()
