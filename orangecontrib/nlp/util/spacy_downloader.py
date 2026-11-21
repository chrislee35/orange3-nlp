import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

_RELEASES = "https://github.com/explosion/spacy-models/releases/download"


class SpaCyDownloader:
    @staticmethod
    def model_name(language_code: str) -> str | None:
        SPACY_LANGUAGE_MODELS: dict[str, str] = {
            "ca": "ca_core_news_sm",
            "zh": "zh_core_web_sm",
            "da": "da_core_news_sm",
            "nl": "nl_core_news_sm",
            "en": "en_core_web_sm",
            "fr": "fr_core_news_sm",
            "de": "de_core_news_sm",
            "el": "el_core_news_sm",
            "it": "it_core_news_sm",
            "ja": "ja_core_news_sm",
            "lt": "lt_core_news_sm",
            "mk": "mk_core_news_sm",
            "nb": "nb_core_news_sm",
            "pl": "pl_core_news_sm",
            "pt": "pt_core_news_sm",
            "ro": "ro_core_news_sm",
            "ru": "ru_core_news_sm",
            "es": "es_core_news_sm",
            "sl": "sl_core_news_sm",
            "sv": "sv_core_news_sm",
            "uk": "uk_core_news_sm",
            "xx": "xx_ent_wiki_sm",  # Multi-language model
        }

        language_code = language_code.lower()
        return SPACY_LANGUAGE_MODELS.get(language_code)

    @staticmethod
    def require_model_name(language_code: str) -> str:
        model_name = SpaCyDownloader.model_name(language_code)
        if model_name is None:
            raise ValueError(f"No spaCy model is available for language '{language_code}'")
        return model_name

    @staticmethod
    def download_language(language_code: str) -> bool:
        model_name = SpaCyDownloader.model_name(language_code)
        if model_name is None:
            return False
        SpaCyDownloader.download(model_name)
        return True

    @staticmethod
    def models_dir() -> Path:
        """Per-user, always-writable folder holding models downloaded by this add-on."""
        import platformdirs

        return Path(platformdirs.user_data_dir("orange3-nlp", appauthor=False)) / "spacy-models"

    @staticmethod
    def local_model_path(model_name: str) -> Path | None:
        """Newest downloaded copy of ``model_name`` in the per-user folder, if any."""
        root = SpaCyDownloader.models_dir()
        if not root.is_dir():
            return None
        found = sorted(
            (d for d in root.iterdir() if d.is_dir() and d.name.startswith(model_name + "-")),
            key=lambda d: d.name,
        )
        return found[-1] if found else None

    @staticmethod
    def _wheel_url(model_name: str) -> tuple[str, str]:
        from spacy.cli.download import get_compatibility, get_version

        version = get_version(model_name, get_compatibility())
        full = f"{model_name}-{version}"
        return full, f"{_RELEASES}/{full}/{full}-py3-none-any.whl"

    @staticmethod
    def _extract_wheel(data: bytes, model_name: str, full: str, target: Path) -> None:
        """Unpack ``<model>/<model>-<version>/`` from a model wheel into ``target``."""
        prefix = f"{model_name}/{full}/"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=target.parent, prefix=".partial-"))
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                members = [m for m in zf.namelist() if m.startswith(prefix) and not m.endswith("/")]
                if not members:
                    raise ValueError(f"{full} wheel does not contain {prefix}")
                for m in members:
                    dest = (tmp / m[len(prefix):]).resolve()
                    if tmp.resolve() not in dest.parents:
                        raise ValueError(f"Unsafe path in wheel: {m}")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
            if target.exists():
                shutil.rmtree(target)
            os.replace(tmp, target)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def download(model_name: str) -> bool:
        """Download ``model_name`` into the per-user folder (no pip, no admin rights)."""
        try:
            import requests

            full, url = SpaCyDownloader._wheel_url(model_name)
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            SpaCyDownloader._extract_wheel(
                response.content, model_name, full, SpaCyDownloader.models_dir() / full
            )
        except Exception:
            return False
        return True

    @staticmethod
    def load(model_name: str) -> Any:
        """Load a spaCy model: installed package first, then per-user copy, then download.

        Raises OSError with instructions if the model cannot be obtained.
        """
        import spacy

        try:
            return spacy.load(model_name)
        except OSError:
            pass
        path = SpaCyDownloader.local_model_path(model_name)
        if path is None and SpaCyDownloader.download(model_name):
            path = SpaCyDownloader.local_model_path(model_name)
        if path is None:
            raise OSError(
                f"spaCy model '{model_name}' is not installed and could not be downloaded "
                f"(are you offline?). Download '{model_name}' from "
                f"https://github.com/explosion/spacy-models/releases and unpack it into "
                f"{SpaCyDownloader.models_dir()}, or run: python -m spacy download {model_name}"
            )
        return spacy.load(path)
