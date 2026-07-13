import torch

from model.tp2_modules import DeformableTransformer1D, TemporalFeaturePyramid


def test_temporal_feature_pyramid_shapes():
    batch, length, channels = 2, 64, 32
    src = torch.randn(batch, length, channels)
    valid_mask = torch.ones(batch, length, dtype=torch.bool)
    valid_mask[1, 50:] = False

    pyramid = TemporalFeaturePyramid(
        d_model=channels,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        stem_layers=1,
        branch_layers=3,
        downsample_rate=2,
        window_size=9,
    )
    levels, masks = pyramid(src, valid_mask)
    assert [x.shape[1] for x in levels] == [64, 32, 16, 8]
    assert [m.shape[1] for m in masks] == [64, 32, 16, 8]


def test_deformable_transformer_forward_backward():
    batch, length, channels = 2, 64, 32
    num_queries = 10
    src = torch.randn(batch, length, channels, requires_grad=True)
    valid_mask = torch.ones(batch, length, dtype=torch.bool)
    valid_mask[1, 50:] = False

    transformer = DeformableTransformer1D(
        d_model=channels,
        nhead=4,
        num_queries=num_queries,
        num_encoder_layers=2,
        num_decoder_layers=3,
        dim_feedforward=64,
        dropout=0.0,
        num_feature_levels=4,
        enc_n_points=4,
        dec_n_points=4,
        fpn_stem_layers=1,
        fpn_branch_layers=3,
        fpn_downsample_rate=2,
        fpn_window_size=9,
    )
    hs, references, memory, memory_global = transformer(
        src,
        ~valid_mask,
        torch.randn(num_queries, 2),
        torch.zeros_like(src),
        torch.randn(batch, 1, channels),
        torch.randn(batch, 1, channels),
    )

    assert hs.shape == (3, batch, num_queries, channels)
    assert references.shape == (3, batch, num_queries, 2)
    assert memory.shape == (batch, length, channels)
    assert memory_global.shape == (batch, channels)

    loss = hs.square().mean() + references.square().mean() + memory.square().mean()
    loss.backward()
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()
