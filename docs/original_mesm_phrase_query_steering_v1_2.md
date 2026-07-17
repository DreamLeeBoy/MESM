# Original MESM + Phrase Query Steering v1.2

This branch starts from the original MESM `main` commit:

```text
8501bdea711b7897952ed2bdacf2883665c0950b
```

It deliberately does **not** use the MESM-v2 architecture.

## Retained original MESM components

- frozen CLIP text encoder
- FW-MESM
- SS-MESM
- original T2V encoder
- original Transformer encoder/decoder
- original two-dimensional DAB reference queries
- original saliency negative-query branch
- original matcher and losses

## Added components

- exact CLIP-BPE action/object masks from the audited v1.2 sidecar
- action-residual phrase composition: `z = Normalize(LayerNorm(a + MLP([a; o])))`
- norm-preserving Query Steering immediately before the original first T2V layer
- training-only same-video Level-1 competitor margin loss

## Explicitly absent

- Pathformer
- path-balance loss
- MESM-v2 decoder text cross-attention
- MESM-v2 extra learnable query-content embedding
- MESM-v2 background-aware ranking

## Sidecar files

Place these files under:

```text
data/charades/annotations/semantic_audit_v1_2/
├── charades_sta_train_semantic_final_v1_2.jsonl
└── charades_sta_test_semantic_final_v1_2.jsonl
```

Expected line counts:

```text
12408 train
3720 test
```

The test sidecar must be sanitized so it contains no temporal target or candidate information.

## Validation

```bash
python -m pytest -q tests/test_phrase_steering.py
python tools/smoke_test_phrase_cuda.py
```

The CUDA smoke test uses two video groups because the original MESM cross-video negative branch requires an out-of-video text candidate.

## Training

```bash
python train_phrase.py \
  --config_file ./config/charades/C+SF_C_original_MESM_phrase_steering_v1_2.json
```

## Evaluation

Use the original MESM evaluation arguments through the wrapper:

```bash
python eval_phrase.py ...
```

## Main hyperparameters

```json
{
  "enable_phrase_steering": true,
  "phrase_steering_layer": 0,
  "phrase_steering_init_logit": -3.0,
  "phrase_mlp_dropout": 0.1,
  "loss_competitor_coef": 0.1,
  "competitor_margin": 0.2
}
```
