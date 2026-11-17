from AnyQt.QtWidgets import QBoxLayout, QLayout, QWidget


def layout_of(widget: QWidget) -> QLayout:
    """Return a widget's layout, which Qt types as Optional but Orange always sets."""
    layout = widget.layout()
    assert layout is not None, "widget has no layout"
    return layout


def box_layout_of(widget: QWidget) -> QBoxLayout:
    """Return a widget's layout as a box layout (Orange's control and main areas use one)."""
    layout = widget.layout()
    assert isinstance(layout, QBoxLayout), "widget does not have a box layout"
    return layout
