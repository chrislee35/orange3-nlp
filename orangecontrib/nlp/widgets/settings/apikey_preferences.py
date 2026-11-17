from orangecanvas.application.settings import UserSettingsDialog, FormLayout
from typing import Any

from AnyQt.QtWidgets import QWidget, QFormLayout, QLineEdit
from orangecontrib.nlp.util import apikeys


class APIKeySettings(UserSettingsDialog):
    name = "API Keys"  # Tab name in Preferences
    services = ["openai", "gemini"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        w = self.widget(0)  # 'General' tab
        assert w is not None
        layout = w.layout()
        assert isinstance(layout, QFormLayout)

        tab = QWidget()
        self.addTab(tab, self.tr("API Keys"), toolTip="API Keys for remote services")
        form = FormLayout()
        for service in APIKeySettings.services:
            field = QLineEdit()
            field.setText(apikeys.get_api_key(service))
            field.editingFinished.connect(lambda s=service, f=field: apikeys.set_api_key(s, f.text()))
            form.addRow(service.capitalize() + " API Key:", field)

        tab.setLayout(form)
