# MESM v2: FW-CDL

MESM v2 moves Complementary Learning Exploiting Class Diversities (CDL)
from the segment-sentence space to forward word reconstruction. The model
architecture and the original localization, classification, saliency, `rec_ss`,
and `rec_fw` losses remain unchanged.

FW-CDL is evaluated only for valid word positions that were actually masked and
whose full-vocabulary reconstruction prediction is wrong. For each such anchor,
its ground-truth POS selects an independent verb, noun, or pronoun candidate
universe. Candidates are the unique same-POS masked ground-truth classes in the
current batch; when that set has fewer than two classes, the loss uses the
classifier-supported global vocabulary for that POS.

Frozen MESM word representations define cosine dissimilarity
`d(y,j) = 1 - cos(e_y, e_j)`. The teacher and student distributions are:

- `T_j = softmax(-d(y,j) / tau)`
- `P_j = softmax(recfw_words_logit[j] / tau)`

The additional loss is `L_fw_cdl = KL(T || P)`, and the complete FW contribution
is weighted through the existing criterion mechanism as
`L_fw_total = L_fw + lambda_fw_cdl * L_fw_cdl`. Defaults are `tau = 0.07` and
`lambda_fw_cdl = 0.1`.
