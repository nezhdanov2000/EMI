"""
Example/demo assets bundled with this repository — not part of the
installable `vsf` library.

Making this an explicit package (rather than relying on Python 3.3+
implicit namespace packages) keeps `from examples.mushroom_demo import
MUSHROOM_TRANSLATIONS` (used by `server.py`) unambiguous regardless of
what else might be on `sys.path`.
"""
