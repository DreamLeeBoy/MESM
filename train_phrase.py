"""Train MESM-v2 with phrase-level query steering enabled by config."""

import runpy

from phrase_steering import install_phrase_steering


if __name__ == "__main__":
    install_phrase_steering()
    runpy.run_module("train", run_name="__main__")
