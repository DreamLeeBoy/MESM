"""Evaluate TP²-MESM while reusing the official MESM evaluation code."""
import eval as mesm_eval
from runner_tp2 import build_model, build_criterion

mesm_eval.build_model = build_model
mesm_eval.build_criterion = build_criterion

if __name__ == "__main__":
    mesm_eval.inference()
