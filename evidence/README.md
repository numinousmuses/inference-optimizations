# Evidence

[`gate/`](gate/) holds 271 run records from the serving and accuracy gates (see [docs/methodology.md](../docs/methodology.md)). Each file name is `<model>_<lane>_<arm>_<timestamp>.json`.

Each serving record has the image digest, the config digest, flags, env, the patch series, per-repeat timings (steady and warm aggregate tok/s, per-user tok/s, TTFT percentiles), the KV pool size, speculative acceptance and probe results. Accuracy records (`*_accuracy_*`) have per-arm KL, top-1, p99 |dlogprob| and GSM8K, with the stock-vs-stock noise floor.

Where to start:

| Claim | Files |
|---|---|
| DeepSeek V4 Pro +61 to 65% vs stock (8xB200) | `deepseek-v4-pro_standard_reg_final_pro_*`, `*v20_stock*`, `*megamoe_w4a4_bal*` |
| DeepSeek V4 Pro on 8xB300 | `deepseek-v4-pro_standard_B300:8_final_pro_*` |
| DeepSeek V4 Pro accuracy | `deepseek-v4-pro_standard_accuracy_2026092*` |
| GLM-5.3 composed recipe and fused DP-MoE | `glm-5.3_standard_reg_final_glm_*`, `*glm_v20_dpmoe*` |
| GLM-5.3 4-bit KV | `*kv4glm*`, `*kv4mla*` |
| RTX 4-bit KV (Triton backend) | `*kv4tri*` |
| DeepSeek V4 Pro 4-bit KV | `*kv4pro*`, `deepseek-v4-pro_standard_reg_final_pro_kv4*` |
