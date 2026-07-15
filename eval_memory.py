import eval as v2_eval
from memory_entrypoint import build_model


# Rebuild the same adapter before loading the memory-augmented checkpoint.
v2_eval.build_model = build_model


if __name__ == "__main__":
    v2_eval.inference()
