# Phrase Query Steering v1.2 for MESM-v2

This implementation adds an action-object phrase token, target-only visual query steering, and a training-only same-video competitor loss without modifying the original MESM-v2 source files.

## Branch

```text
feat/phrase-query-steering-v1-2
```

The original `dataset/`, `model/`, `runner.py`, `train.py`, and `eval.py` files remain unchanged. Integration is installed at runtime by `train_phrase.py` and `eval_phrase.py`.

## Required semantic files

Place the v1.2 training sidecar at:

```text
data/charades/annotations/semantic_audit_v1_2/
└── charades_sta_train_semantic_final_v1_2.jsonl
```

The candidate CSV is useful for auditing but is not read during training because candidate IDs are already stored in each sidecar record:

```text
level1_candidate_pairs_final_v1_2.csv
```

### Generate the test sidecar

Query Steering is used during validation and inference, so a test sidecar is required. Generate it from text using the same v1.2 parser and official CLIP BPE file:

```bash
python ~/下载/charades_semantic_audit_v1_2/build_charades_semantic_sidecar_v1_2.py \
  --train-path ./data/charades/annotations/charades_sta_test.txt \
  --bpe-path ./pretrained_models/bpe_simple_vocab_16e6.txt.gz \
  --output-dir /tmp/charades_test_semantic_v1_2 \
  --max-words-l 16 \
  --iou-threshold 0.3 \
  --sample-size 100 \
  --seed 2019
```

The builder keeps its historical `train` filename even when the supplied input is the test annotation. Sanitize and rename it before evaluation:

```bash
python tools/sanitize_charades_test_sidecar.py \
  --input /tmp/charades_test_semantic_v1_2/charades_sta_train_semantic_final.jsonl \
  --output ./data/charades/annotations/semantic_audit_v1_2/charades_sta_test_semantic_final_v1_2.jsonl
```

The sanitizer removes temporal windows and candidate lists. The inference sidecar therefore contains only text-derived phrase metadata and exact CLIP BPE positions.

## Regression tests

```bash
pytest -q tests/test_phrase_steering.py
```

The tests cover:

- invalid phrase rows are exact no-ops;
- padding frames are not modified;
- steering preserves per-frame feature norms;
- local same-video competitor indices become correct flattened batch indices;
- competitor loss ignores invalid rows.

## Training

```bash
python train_phrase.py \
  --config_file ./config/charades/C+SF_C_phrase_steering_v1_2.json
```

The full configuration uses:

```text
enable_phrase_steering = true
phrase_steering_layer = 0
phrase_steering_init_logit = -3.0
phrase_mlp_dropout = 0.1
loss_competitor_coef = 0.1
competitor_margin = 0.2
max_gather_size = -1
```

All models are trained from scratch. The existing CLIP text encoder remains frozen, while `input_txt_proj`, the phrase MLP, the steering scale, and the original MESM-v2 trainable modules are optimized jointly.

## Evaluation

Use the standard evaluation configuration but launch through `eval_phrase.py`:

```bash
python eval_phrase.py \
  --config_file ./config/charades/C+SF_C_eval.json
```

`TestOptions` loads the phrase settings from the training run's saved `opt.json`. Update `trained_result_dir` in the evaluation config to the phrase-steering experiment directory.

## Ablations

The same runtime supports the minimum ablation matrix without code changes.

### Baseline path through the new launcher

```json
"enable_phrase_steering": false,
"loss_competitor_coef": 0.0,
"require_semantic_sidecar": false
```

### Steering only

```json
"enable_phrase_steering": true,
"loss_competitor_coef": 0.0
```

### Competitor loss only

```json
"enable_phrase_steering": false,
"loss_competitor_coef": 0.1
```

### Full method

```json
"enable_phrase_steering": true,
"loss_competitor_coef": 0.1
```

## Data flow

```text
CLIP contextualized BPE features
  ├── action-mask mean pooling
  ├── object-mask mean pooling
  └── action residual + MLP([action; object])
          ↓
     phrase token [B, 256]
          ├── positive T2V visual-content steering, first layer only
          └── training-only GT-window competitor margin loss
```

The existing out-of-video negative saliency branch receives no phrase token and is unchanged in function. Competitor queries are never required at inference.
