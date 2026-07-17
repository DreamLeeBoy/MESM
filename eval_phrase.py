"""Evaluate original MESM with phrase construction and Query Steering."""

import runpy

from phrase_steering import install_phrase_steering


if __name__ == "__main__":
    install_phrase_steering()
    runpy.run_module("eval", run_name="__main__")
