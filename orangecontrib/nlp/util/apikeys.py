# orangecontrib/nlp/util/apikeys.py

from AnyQt.QtCore import QSettings

_ORANGE_SETTINGS_GROUP = "nlp/api_keys"

def _settings():
    return QSettings()

def set_api_key(service_name: str, api_key: str):
    s = _settings()
    s.beginGroup(_ORANGE_SETTINGS_GROUP)
    s.setValue(service_name, api_key)
    s.endGroup()

def get_api_key(service_name: str) -> str:
    s = _settings()
    s.beginGroup(_ORANGE_SETTINGS_GROUP)
    key = s.value(service_name, "", type=str)
    s.endGroup()
    return key or ""

def available_services() -> list[str]:
    s = _settings()
    s.beginGroup(_ORANGE_SETTINGS_GROUP)
    keys = list(map(str, s.allKeys()))
    s.endGroup()
    return keys
