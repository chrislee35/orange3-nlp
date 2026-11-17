#!/usr/bin/env python3
import os

from setuptools import setup

DOC_BUILD = os.path.join("doc", "_build", "html")
DOC_TARGET = os.path.join("help", "orange3-nlp")


def documentation_files() -> list[tuple[str, list[str]]]:
    """Return the built Sphinx HTML as data files, preserving the directory layout.

    Orange looks for add-on help under ``<data>/help/<addon>`` (see ``WIDGET_HELP_PATH``).
    The list is empty when the documentation has not been built.
    """
    files = []
    for directory, _, names in os.walk(DOC_BUILD):
        if not names:
            continue
        relative = os.path.relpath(directory, DOC_BUILD)
        target = DOC_TARGET if relative == "." else os.path.join(DOC_TARGET, relative)
        files.append((target, [os.path.join(directory, name) for name in names]))
    return files


if __name__ == "__main__":
    setup(data_files=documentation_files())
