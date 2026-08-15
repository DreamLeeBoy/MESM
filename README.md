Experimental branch: rec_ss_hard_negative_comp

Description:
This branch extends the original rec_ss similarity learning with
hardness-aware complementary learning.

Motivation:
Existing contrastive learning treats negative samples equally.
Inspired by complementary learning, we model the diversity among
negative samples and introduce hardness-aware negative weighting.

Modification:
- Keep original MESM architecture.
- Keep Hungarian matching.
- Keep learnable span prediction unchanged.
- Add complementary supervision based on negative sample relationships.

Loss:
L_total = L_MESM + lambda * L_comp

where L_comp aligns the predicted negative distribution with the
hardness-aware complementary distribution.
