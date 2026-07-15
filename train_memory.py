import train as v2_train
from memory_entrypoint import build_model


# Reuse the original v2 data, criterion, optimizer, logging and evaluation code.
v2_train.build_model = build_model


if __name__ == "__main__":
    v2_train.train()
