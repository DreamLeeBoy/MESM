from runner import build_model as build_v2_model
from model.memory_adapter import attach_dynamic_temporal_memory


def build_model(args, vocab=None):
    model = build_v2_model(args, vocab)
    return attach_dynamic_temporal_memory(model, args)
