"""Train TP²-MESM while reusing the official MESM training loop unchanged."""
import train as mesm_train
from runner_tp2 import build_model, build_criterion

mesm_train.build_model = build_model
mesm_train.build_criterion = build_criterion

if __name__ == "__main__":
    mesm_train.train()
