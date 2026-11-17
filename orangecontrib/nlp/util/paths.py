import os

from Orange.misc.environ import data_dir_base


def orange_data_dir(*parts: str) -> str:
    """Return a path under Orange's data directory (``<data_dir_base>/Orange/...``)."""
    base = data_dir_base() or os.path.expanduser("~")
    return os.path.join(base, "Orange", *parts)
