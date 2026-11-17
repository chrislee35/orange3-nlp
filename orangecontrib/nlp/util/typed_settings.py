from typing import Any, TypeVar

from Orange.widgets.settings import Setting

T = TypeVar("T")


def setting(default: T, **kwargs: Any) -> T:
    """Declare an Orange widget setting while keeping the value's static type.

    Orange's ``Setting`` descriptor is untyped, so type checkers see a ``Setting``
    object where the widget actually holds the value. At runtime this returns the
    same ``Setting`` instance that Orange's settings handler looks for.
    """
    return Setting(default, **kwargs)  # type: ignore[return-value]
