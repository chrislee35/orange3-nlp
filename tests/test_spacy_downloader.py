import io
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from orangecontrib.nlp.util.spacy_downloader import SpaCyDownloader as D


def _wheel(name="xx_test_sm", version="3.8.0", extra=None):
    full = f"{name}-{version}"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{name}/__init__.py", "")
        zf.writestr(f"{name}/{full}/meta.json", "{}")
        zf.writestr(f"{name}/{full}/ner/model", "x")
        for n in extra or []:
            zf.writestr(n, "evil")
    return name, full, buf.getvalue()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "models_dir", staticmethod(lambda: tmp_path / "models"))
    return tmp_path / "models"


def test_extract_wheel(home):
    name, full, data = _wheel()
    D._extract_wheel(data, name, full, home / full)
    assert (home / full / "meta.json").exists()
    assert (home / full / "ner" / "model").exists()
    assert not (home / full / "__init__.py").exists()
    assert D.local_model_path(name) == home / full
    assert not [p for p in home.iterdir() if p.name.startswith(".partial")]


def test_extract_rejects_traversal(home):
    name, full, data = _wheel(extra=[f"xx_test_sm/xx_test_sm-3.8.0/../../evil"])
    with pytest.raises(ValueError):
        D._extract_wheel(data, name, full, home / full)
    assert not (home / full).exists()
    assert not (home.parent / "evil").exists()


def test_download_success_and_failure(home):
    name, full, data = _wheel()
    resp = mock.Mock(content=data, raise_for_status=lambda: None)
    with mock.patch.object(D, "_wheel_url", return_value=(full, "http://x")), \
            mock.patch("requests.get", return_value=resp):
        assert D.download(name)
    assert D.local_model_path(name) == home / full
    with mock.patch.object(D, "_wheel_url", side_effect=OSError("offline")):
        assert not D.download(name)


def test_load_order(home):
    import spacy
    sentinel = object()
    with mock.patch.object(spacy, "load", return_value=sentinel) as load:
        assert D.load("xx_test_sm") is sentinel
        load.assert_called_once_with("xx_test_sm")

    # not installed, no local copy, download fails -> clear OSError
    with mock.patch.object(spacy, "load", side_effect=OSError), \
            mock.patch.object(D, "download", return_value=False):
        with pytest.raises(OSError, match="xx_test_sm"):
            D.load("xx_test_sm")

    # not installed, downloaded copy used by path
    name, full, data = _wheel()
    D._extract_wheel(data, name, full, home / full)
    calls = []

    def fake_load(arg):
        calls.append(arg)
        if arg == name:
            raise OSError
        return sentinel

    with mock.patch.object(spacy, "load", side_effect=fake_load), \
            mock.patch.object(D, "download") as dl:
        assert D.load(name) is sentinel
    dl.assert_not_called()
    assert calls == [name, home / full]
