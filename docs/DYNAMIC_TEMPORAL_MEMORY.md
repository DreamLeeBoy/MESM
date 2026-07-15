# Dynamic Temporal Memory on MESM v2

This experiment is implemented as an opt-in overlay on the unchanged `v2`
training pipeline.

## Method

The plugin stores a bounded bank of normalized temporal prototypes. For every
T2V output it:

1. pools the current text and video into a retrieval query;
2. retrieves top-k prototypes by cosine similarity;
3. applies a text- and memory-conditioned frame-wise residual;
4. selects the most text-relevant frames from the positive branch;
5. EMA-merges similar prototypes or inserts novel ones into a bounded FIFO bank.

MESM v2 calls the T2V encoder once with the matching sentence and once with a
batch-shuffled negative sentence. Both branches retrieve memory, but only the
first positive call may update the bank. `model.eval()` freezes memory updates.
The bank is a registered buffer and is saved in the model checkpoint.

## Train

```bash
python train_memory.py \
  --config_file config/charades/C+SF_C_temporal_memory.json
```

## Evaluate

Use the result directory produced above:

```bash
python eval_memory.py \
  --trained_result_dir results/charades/<experiment-directory>
```

The standard `train.py` and original configs remain the unmodified v2 baseline.
Training from an old baseline checkpoint is not enabled in this first version,
because strict loading correctly reports the new memory keys as missing.
