import json
import unittest
from typing import Any
from unittest import mock

from orangecontrib.text import Corpus
from orangecontrib.nlp.util.text_generation import (
    GenerationSettings,
    HuggingFaceGenerator,
    OllamaGenerator,
    build_output,
    make_generator,
    make_jobs,
    render_prompt,
    row_values,
)


class FakeResponse:
    def __init__(self, lines: list[dict[str, Any]], status_code: int = 200) -> None:
        self.lines = [json.dumps(line).encode() for line in lines]
        self.status_code = status_code
        self.text = "boom"

    def iter_lines(self) -> list[bytes]:
        return self.lines

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class TestRenderPrompt(unittest.TestCase):
    def test_replaces_known_placeholders(self) -> None:
        text, unknown = render_prompt("Summarize {Title}: {text}", {"Title": "T", "text": "body"})
        self.assertEqual(text, "Summarize T: body")
        self.assertEqual(unknown, [])

    def test_unknown_placeholders_are_kept_and_reported_once(self) -> None:
        text, unknown = render_prompt("{a} {a} {text}", {"text": "x"})
        self.assertEqual(text, "{a} {a} x")
        self.assertEqual(unknown, ["a"])

    def test_json_braces_survive(self) -> None:
        text, unknown = render_prompt('Return {"key": 1} for {text}', {"text": "x"})
        self.assertEqual(text, 'Return {"key": 1} for x')
        self.assertEqual(unknown, ['"key": 1'])


class TestSettings(unittest.TestCase):
    def test_seed_differs_per_sample(self) -> None:
        settings = GenerationSettings(seed=10)
        self.assertEqual([settings.seed_for(i) for i in range(3)], [10, 11, 12])

    def test_no_seed(self) -> None:
        self.assertIsNone(GenerationSettings().seed_for(2))

    def test_jobs_order(self) -> None:
        jobs = make_jobs(["a", "b"], 2)
        self.assertEqual([(j.row, j.sample, j.prompt) for j in jobs], [(0, 0, "a"), (0, 1, "a"), (1, 0, "b"), (1, 1, "b")])


class TestOllamaGenerator(unittest.TestCase):
    def test_streams_and_sends_options(self) -> None:
        response = FakeResponse([{"response": "Hello"}, {"response": " world"}, {"done": True}])
        with mock.patch("requests.post", return_value=response) as post:
            pieces: list[str] = []
            generator = OllamaGenerator("localhost", 11434, "phi4-mini")
            text = generator.generate(
                "Hi", GenerationSettings(temperature=0.2, max_new_tokens=5, top_p=0.8, seed=3), 1, pieces.append, lambda: False
            )
        self.assertEqual(text, "Hello world")
        self.assertEqual(pieces, ["Hello", " world"])
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertEqual(kwargs["json"]["prompt"], "Hi")
        self.assertEqual(kwargs["json"]["options"], {"temperature": 0.2, "num_predict": 5, "top_p": 0.8, "seed": 4})

    def test_no_seed_option_when_random(self) -> None:
        options = OllamaGenerator("h", 1, "m").options(GenerationSettings(), 0)
        self.assertNotIn("seed", options)

    def test_http_error(self) -> None:
        with mock.patch("requests.post", return_value=FakeResponse([], status_code=404)):
            with self.assertRaises(RuntimeError):
                OllamaGenerator("h", 1, "m").generate("x", GenerationSettings(), 0, None, lambda: False)

    def test_error_message_in_stream(self) -> None:
        with mock.patch("requests.post", return_value=FakeResponse([{"error": "model not found"}])):
            with self.assertRaisesRegex(RuntimeError, "model not found"):
                OllamaGenerator("h", 1, "m").generate("x", GenerationSettings(), 0, None, lambda: False)

    def test_cancel_stops_early(self) -> None:
        with mock.patch("requests.post", return_value=FakeResponse([{"response": "a"}, {"response": "b"}])):
            text = OllamaGenerator("h", 1, "m").generate("x", GenerationSettings(), 0, None, lambda: True)
        self.assertEqual(text, "")


class TestFactory(unittest.TestCase):
    def test_backends(self) -> None:
        self.assertIsInstance(make_generator("Ollama", "m"), OllamaGenerator)
        self.assertIsInstance(make_generator("Hugging Face", "gpt2"), HuggingFaceGenerator)
        with self.assertRaises(ValueError):
            make_generator("nope", "m")


class TestBuildOutput(unittest.TestCase):
    def test_without_input(self) -> None:
        jobs = make_jobs(["P"], 2)
        out = build_output(None, jobs, ["x", "y"], ["P"])
        self.assertEqual(len(out), 2)
        self.assertEqual([str(out[i]["Generated Text"]) for i in range(2)], ["x", "y"])
        self.assertEqual([float(out[i]["Sample"]) for i in range(2)], [1.0, 2.0])
        self.assertEqual([str(f) for f in out.text_features], ["Generated Text"])

    def test_with_input_keeps_columns_and_repeats_rows(self) -> None:
        corpus = Corpus("book-excerpts")[:2]
        prompts = [f"p{i}" for i in range(2)]
        jobs = make_jobs(prompts, 2)
        out = build_output(corpus, jobs, ["a", "b", "c", "d"], prompts)
        self.assertEqual(len(out), 4)
        self.assertEqual([str(out[i]["Prompt"]) for i in range(4)], ["p0", "p0", "p1", "p1"])
        first_title = [m for m in corpus.domain.metas if m.name != "Text"]
        for var in first_title:
            self.assertEqual(str(out[0][var.name]), str(corpus[0][var]))
            self.assertEqual(str(out[1][var.name]), str(corpus[0][var]))
            self.assertEqual(str(out[2][var.name]), str(corpus[1][var]))
        self.assertEqual(len(out.domain.metas), len(corpus.domain.metas) + 3)

    def test_row_values(self) -> None:
        corpus = Corpus("book-excerpts")[:1]
        values = row_values(corpus, 0)
        self.assertEqual(values["text"], corpus.documents[0])
        for var in (*corpus.domain.variables, *corpus.domain.metas):
            self.assertIn(var.name, values)


if __name__ == "__main__":
    unittest.main()
