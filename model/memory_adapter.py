import types

import torch

from .dynamic_temporal_memory import DynamicTemporalMemory


def attach_dynamic_temporal_memory(model, args):
    """Attach the memory plugin to an unmodified MESM v2 model.

    The adapter patches only runtime methods, so the original ``v2`` model and
    T2V parameter names remain unchanged. A separate ``temporal_memory`` module
    is registered on MESM and therefore saved in normal checkpoints.
    """
    if not getattr(args, "use_temporal_memory", False):
        return model
    if hasattr(model, "temporal_memory"):
        return model

    memory = DynamicTemporalMemory(
        hidden_dim=args.hidden_dim,
        capacity=getattr(args, "temporal_memory_capacity", 2048),
        topk=getattr(args, "temporal_memory_topk", 5),
        min_entries=getattr(args, "temporal_memory_min_entries", 32),
        temperature=getattr(args, "temporal_memory_temperature", 0.07),
        momentum=getattr(args, "temporal_memory_momentum", 0.9),
        merge_threshold=getattr(args, "temporal_memory_merge_threshold", 0.95),
        residual_scale=getattr(args, "temporal_memory_residual_scale", 0.2),
        prototype_topk_frames=getattr(args, "temporal_memory_prototype_topk_frames", 4),
        min_text_similarity=getattr(args, "temporal_memory_min_text_similarity", 0.0),
    ).to(args.device)
    model.add_module("temporal_memory", memory)

    original_t2v_forward = model.t2v_encoder.forward
    original_model_forward = model.forward
    model._temporal_memory_update_pending = False
    model._temporal_memory_last_state = None

    def t2v_forward_with_memory(
        encoder_self,
        src_txt,
        src_vid,
        src_txt_mask=None,
        src_txt_key_padding_mask=None,
        pos_txt=None,
        src_vid_mask=None,
        src_vid_key_padding_mask=None,
        pos_vid=None,
        **kwargs
    ):
        encoded = original_t2v_forward(
            src_txt=src_txt,
            src_vid=src_vid,
            src_txt_mask=src_txt_mask,
            src_txt_key_padding_mask=src_txt_key_padding_mask,
            pos_txt=pos_txt,
            src_vid_mask=src_vid_mask,
            src_vid_key_padding_mask=src_vid_key_padding_mask,
            pos_vid=pos_vid,
            **kwargs
        )

        if src_txt_key_padding_mask is None:
            text_valid = torch.ones(
                src_txt.shape[:2], dtype=torch.bool, device=src_txt.device
            )
        else:
            text_valid = ~src_txt_key_padding_mask.bool()
        if src_vid_key_padding_mask is None:
            video_valid = torch.ones(
                encoded.shape[:2], dtype=torch.bool, device=encoded.device
            )
        else:
            video_valid = ~src_vid_key_padding_mask.bool()

        text_context = memory.masked_mean(src_txt, text_valid)
        enhanced, state = memory(encoded, video_valid, text_context)
        model._temporal_memory_last_state = state

        # MESM v2 calls T2V twice: positive text first, batch-shuffled negative
        # text second. Only the first call is allowed to evolve the bank.
        if model.training and model._temporal_memory_update_pending:
            candidates = memory.build_pseudo_prototypes(
                encoded.detach(), video_valid, text_context.detach()
            )
            memory.update(candidates)
            model._temporal_memory_update_pending = False
        return enhanced

    def model_forward_with_memory(model_self, *forward_args, **forward_kwargs):
        model._temporal_memory_update_pending = bool(model.training)
        output = original_model_forward(*forward_args, **forward_kwargs)
        state = model._temporal_memory_last_state
        if state is not None:
            output.update({
                "memory_size": state["size"].detach(),
                "memory_max_similarity": state["max_similarity"].detach(),
                "memory_uncertainty": state["uncertainty"].detach(),
                "memory_gate_mean": state["gate_mean"].detach(),
            })
        return output

    model.t2v_encoder.forward = types.MethodType(
        t2v_forward_with_memory, model.t2v_encoder
    )
    model.forward = types.MethodType(model_forward_with_memory, model)
    return model
