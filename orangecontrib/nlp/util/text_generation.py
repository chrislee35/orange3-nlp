"""Backends and helpers for the Text Generator widget.

Kept free of Qt so it can be unit tested without a GUI, an Ollama server or a downloaded model.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import requests
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from orangecontrib.text.corpus import Corpus

PLACEHOLDER = re.compile(r"\{([^{}\n]+)\}")

OLLAMA_BACKEND = "Ollama"
HF_BACKEND = "Hugging Face"
BACKENDS = [OLLAMA_BACKEND, HF_BACKEND]
DEFAULT_HF_MODEL = "gpt2"
HF_MODELS = ["gpt2", "distilgpt2"]


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling settings shared by all backends.

    ``seed`` of ``None`` means random. When a seed is set, sample ``i`` of a prompt uses ``seed + i`` so that several
    samples of the same prompt differ from each other but the whole run can be repeated exactly.
    """

    temperature: float = 0.7
    max_new_tokens: int = 100
    top_p: float = 0.9
    seed: int | None = None

    def seed_for(self, sample: int) -> int | None:
        return None if self.seed is None else self.seed + sample


Cancelled = Callable[[], bool]
OnToken = Callable[[str], None]


class Generator(Protocol):
    def generate(self, prompt: str, settings: GenerationSettings, sample: int, on_token: OnToken | None, cancelled: Cancelled) -> str: ...


def render_prompt(template: str, values: Mapping[str, str]) -> tuple[str, list[str]]:
    """Replace ``{name}`` with ``values[name]``.

    Placeholders that are not in ``values`` are left exactly as written (so braces in JSON or code samples survive) and
    their names are returned so the caller can warn about possible typos.
    """
    unknown: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in values:
            return values[name]
        if name not in unknown:
            unknown.append(name)
        return match.group(0)

    return PLACEHOLDER.sub(substitute, template), unknown


class OllamaGenerator:
    """Raw ``/api/generate`` completion (no chat template), streamed."""

    def __init__(self, host: str, port: str | int, model: str, timeout: float = 120) -> None:
        self.url = f"http://{host}:{port}/api/generate"
        self.model = model
        self.timeout = timeout

    def options(self, settings: GenerationSettings, sample: int) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": settings.temperature,
            "num_predict": settings.max_new_tokens,
            "top_p": settings.top_p,
        }
        seed = settings.seed_for(sample)
        if seed is not None:
            options["seed"] = seed
        return options

    def generate(self, prompt: str, settings: GenerationSettings, sample: int, on_token: OnToken | None, cancelled: Cancelled) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": True, "options": self.options(settings, sample)}
        pieces: list[str] = []
        with requests.post(self.url, json=body, stream=True, timeout=self.timeout) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
            for line in response.iter_lines():
                if cancelled():
                    break
                if not line:
                    continue
                message = json.loads(line.decode("utf-8"))
                if "error" in message:
                    raise RuntimeError(f"Ollama error: {message['error']}")
                piece = message.get("response", "")
                if piece:
                    pieces.append(piece)
                    if on_token:
                        on_token(piece)
        return "".join(pieces)


class HuggingFaceGenerator:
    """Local causal language model through ``transformers``. Models are loaded once and cached."""

    _cache: dict[str, tuple[Any, Any]] = {}

    def __init__(self, model: str) -> None:
        self.model_name = model

    def _load(self) -> tuple[Any, Any]:
        if self.model_name not in HuggingFaceGenerator._cache:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            HuggingFaceGenerator._cache[self.model_name] = (tokenizer, model)
        return HuggingFaceGenerator._cache[self.model_name]

    def generate(self, prompt: str, settings: GenerationSettings, sample: int, on_token: OnToken | None, cancelled: Cancelled) -> str:
        import torch

        tokenizer, model = self._load()
        seed = settings.seed_for(sample)
        if seed is not None:
            torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt")
        options: dict[str, Any] = {"max_new_tokens": settings.max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
        if settings.temperature > 0:
            options.update(do_sample=True, temperature=settings.temperature, top_p=settings.top_p)
        else:
            options.update(do_sample=False)
        with torch.no_grad():
            output = model.generate(**inputs, **options)
        text: str = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if on_token and not cancelled():
            on_token(text)
        return text


def make_generator(backend: str, model: str, host: str = "localhost", port: str | int = 11434) -> Generator:
    if backend == OLLAMA_BACKEND:
        return OllamaGenerator(host, port, model)
    if backend == HF_BACKEND:
        return HuggingFaceGenerator(model)
    raise ValueError(f"Unknown backend: {backend}")


@dataclass(frozen=True)
class Job:
    """One generation: a rendered prompt for input row ``row`` (row 0 when there is no input corpus)."""

    row: int
    sample: int
    prompt: str


def make_jobs(prompts: list[str], samples: int) -> list[Job]:
    return [Job(row, sample, prompt) for row, prompt in enumerate(prompts) for sample in range(samples)]


PROMPT_COLUMN = "Prompt"
TEXT_COLUMN = "Generated Text"
SAMPLE_COLUMN = "Sample"


def row_values(corpus: Corpus, row: int) -> dict[str, str]:
    """Values available to the prompt template for one input row: ``text`` plus every column by name."""
    values = {"text": str(corpus.documents[row])}
    for var in (*corpus.domain.variables, *corpus.domain.metas):
        values.setdefault(var.name, var.str_val(corpus[row][var]))
    return values


def build_output(corpus: Corpus | None, jobs: list[Job], texts: list[str], prompts: list[str]) -> Corpus:
    """One output row per job: the input row's columns, the rendered prompt, the generated text and the sample number.

    ``Generated Text`` is set as the corpus's only text feature, so a following *Text Embedder* or *Preprocess Text*
    works on the generations and not on the prompts.
    """
    generated = StringVariable(TEXT_COLUMN)
    new_metas = [StringVariable(PROMPT_COLUMN), generated, ContinuousVariable(SAMPLE_COLUMN)]
    new_values = np.array([[prompts[job.row], text, job.sample + 1] for job, text in zip(jobs, texts)], dtype=object).reshape(len(jobs), 3)
    if corpus is None:
        domain = Domain([], metas=new_metas)
        table = Table.from_numpy(domain, X=np.empty((len(jobs), 0)), metas=new_values)
    else:
        rows = [job.row for job in jobs]
        domain = Domain(corpus.domain.attributes, corpus.domain.class_vars, corpus.domain.metas + tuple(new_metas))
        metas = np.hstack([np.asarray(corpus.metas, dtype=object)[rows], new_values])
        table = Table.from_numpy(domain, X=corpus.X[rows], Y=corpus.Y[rows], metas=metas)
    output = Corpus.from_table(domain, table)
    output.set_text_features([generated])
    return output
