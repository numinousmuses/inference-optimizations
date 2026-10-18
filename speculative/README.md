# Speculative decoding

| Folder | Mechanism | Measured |
|---|---|---|
| [k3-mla-draft/](k3-mla-draft/) | MLA draft models for Kimi K3. Includes draft configs for public HF draft checkpoints | Exact to 1e-9 against the reference on CPU. 30 tests |
| [adaptive-spec-policy/](adaptive-spec-policy/) | Load-adaptive DSpark verify rows, and aux-hidden-state capture under two-batch overlap. Upstream's `AdaptiveSpecPolicy` rejects DSpark under DP | 19 CPU tests. In-engine A/B still to do |
| [gemma-dspark-eagle3/](gemma-dspark-eagle3/) | Fixes that make DSpark and EAGLE3 drafts work on Gemma. The DSpark draft on a tied, scaled embedding lacked `org_vocab_size`/`quant_method`; the EAGLE3 draft ignored `norm_before_residual` | DSpark/DFlash +35 to 53% at 32k context or less on Gemma 4 26B, RTX PRO 6000 |
| [frozen-mtp-draft/](frozen-mtp-draft/) | Frozen-KV MTP draft with no device-to-host sync, a GQA split-K draft decode, and a fused softmax top-1 (top-k = 1 makes softmax a max) | Built and CPU-verified. The top-1 fusion was neutral in the engine |

Speculation depth was also tuned per traffic tier (see [recipes](../recipes/README.md)): EAGLE 5/1/6 wins at 128 sessions or fewer, 3/1/4 at 192 or more. Qwen3.6 MTP at depth 4 gave +30% throughput (acceptance 3.0 to 3.65).
