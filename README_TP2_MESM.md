# TP² modules for MESM

This branch is based directly on the official MESM commit:

```text
8501bdea711b7897952ed2bdacf2883665c0950b
```

The original MESM files and entry points remain unchanged. The TP² variant is additive and uses:

```bash
python train_tp2.py --config_file ./config/charades/C+SF_C_TP2.json
```

For inference:

```bash
python eval_tp2.py --config_file <saved-evaluation-config>
```

## Implemented modules

1. A transformer temporal feature pyramid inserted after `t2v_encoder` and before the DETR localization transformer.
2. A pure-PyTorch one-dimensional multi-scale deformable encoder and decoder.
3. Multi-scale-aware early saliency predictions from both TFPN features and deformable encoder memories.
4. The TP² saliency weighting used by the released implementation:

```text
1.5 × encoder-memory saliency loss + 0.5 × TFPN saliency loss
```

5. Decoder deep supervision for every intermediate layer, using MESM's existing per-layer Hungarian matching and auxiliary loss path.
6. Optional layer-specific classification and span heads, enabled in the provided configuration.

## Preserved MESM behavior

The following official MESM components are reused without rewriting their logic:

- CLIP/GloVe text encoders;
- FW-MESM enhancement encoder;
- T2V alignment encoder;
- SS-MESM reconstruction;
- original positive/negative saliency branch;
- normalized center-width span representation;
- Hungarian matching and MESM localization losses;
- official training and evaluation loops.

The learned MESM global token is retained through a dedicated cross-attention pooling step over the complete multi-scale encoder memory.

## Implementation note

The deformable-attention sampling equations match the one-dimensional TP² implementation. This branch uses the PyTorch reference path based on `grid_sample`, rather than requiring the custom CUDA extension. The mathematical operation is the same, while runtime is slower.

## Validation

The included unit tests verify:

- temporal pyramid lengths `T, T/2, T/4, T/8`;
- decoder hidden-state and reference-point shapes;
- encoder/decoder forward and backward propagation.
