"""Shared test set-up."""
import os

import pytest


@pytest.fixture(autouse=True)
def _no_vision_next_light_note(tmp_path_factory, monkeypatch):
    """Vision Station reads Vision Next's light note (hardware/light.py
    next_owes) from the real vision_next/projects folder. A note left on this
    PC -- Vision Next stopped while it held the light -- changed what the
    light tests saw. Every test gets a note that does not exist unless it
    sets VS_NEXT_LIGHT_HOME itself."""
    if "VS_NEXT_LIGHT_HOME" not in os.environ:
        monkeypatch.setenv("VS_NEXT_LIGHT_HOME",
                           str(tmp_path_factory.getbasetemp() / "no_next_light_home.json"))
