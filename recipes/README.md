# Serving recipes

Tuned SGLang launch configs, one per model and GPU, each measured on real hardware. The raw configs are in [configs/](configs/). Cells read `<context>/b<concurrency> <aggregate tok/s>@<per-user tok/s>`.

| Model | GPU | Quant | Parallelism | Key flags | Headline cell |
|---|---|---|---|---|---|
| nvidia/GLM-5.2-NVFP4 (also serves GLM-5.3) | 8x B200 | NVFP4 weights, FP8 KV | TP8 | `--max-running-requests 512`, EAGLE 3/1/4, `--mem-fraction-static 0.88` | 8k/b320 10,931@34.2; 8k/b192 12,305@64.1; 96k warm prefix c=32 3,548@295 |
| DeepSeek-V4-Pro-0813 | 8x B200 | Own ModelOpt NVFP4 export | DP8 + MegaMoE | `--swa-full-tokens-ratio 0.1`, graphs 256; decode lane: DSpark + mrr 320 | 8k/b256 9,073@35.4; decode lane 13,975@54.6; 46k warm c=128 4,706@67.2 |
| moonshotai/Kimi-K3 | 8x B300 | MXFP4 (native) | TP8 | DSpark (block 7, replayssm), `--mamba-full-memory-ratio 0.95`, mem 0.90, mrr 96 | 8k/b48 4,195@87.4; 524k/b3 254@85 |
| moonshotai/Kimi-K2.7-Code | 8x B200 | INT4, FP8 KV | TP8, DP attention off | EAGLE3 (MLA draft) + `tokenspeed_mla` attention, mrr 128 | 8k/b128 9,047@70.7; c=16 2,466@154; 31k warm c=64 2,791@59.7 |
| nvidia/Gemma-4-26B-A4B-NVFP4 | 1x RTX PRO 6000 | NVFP4, FP8 KV | 1 GPU | `--swa-full-tokens-ratio 0.10 --moe-runner-backend flashinfer_cutlass --mem-fraction-static 0.86 --scheduler-recv-interval 16 --max-running-requests 320` | 8k/b128 4,814@37.6; 131k/b22 719; pool 724k to 3.26M tokens |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | 1x RTX PRO 6000 | NVFP4, FP8 KV | 1 GPU | `--attention-backend flashinfer --chunked-prefill-size 2048`, EAGLE 3/1/4, `--enable-linear-replayssm-spec --max-mamba-cache-size 512 --max-running-requests 96` | 8k/b64 4,954@77.4; 261k/b9 872@97 |
| unsloth/Qwen3.8-27B-NVFP4 (candidate) | 1x RTX PRO 6000 | NVFP4, FP8 KV, bf16 mamba state | 1 GPU | as Qwen3.6, chunk 8192, mamba slots 340, mrr 72 | 8k/b64 3,312@51.8 |

## Rules that transfer to other deployments

- **Hybrid (GDN/KDA) models: always set `--max-mamba-cache-size` explicitly.** You need concurrency x 4 slots with radix cache plus NEXTN, and x 1 with radix off. Qwen3.8 collapsed from 12 sessions (3 tok/s per user) at the default and ran 27 tok/s per user with 80 slots.
- **`--swa-full-tokens-ratio` defaults to 0.8, which wastes 55 GB on Gemma 4.** 0.10 grows the pool from 724k to 3.26M tokens.
- **EAGLE depth trades against batch size.** 5/1/6 wins at 128 or fewer, 3/1/4 at 192 or more, because verify cost scales with draft tokens.
- **DP attention hurts long-prompt warm-prefix traffic.** It splits the radix cache: 30% hits at 31k prompts on Kimi K2.7.
- **`tokenspeed_mla` doubles prefill** over the default MLA backend on Blackwell.
- **`--mamba-radix-cache-strategy extra_buffer`** enables hybrid prefix caching: 99% warm hits on GLM-5.3-Flash.
- **GLM-5.3 on 8xB300 runs best as two 4-GPU replicas.** That beats one 8-GPU DP8 replica by 76% steady and 60% per user.
- **Check the vendor's throughput claims before trusting them.** Two cookbook high-concurrency cells measured at 0.36x and 0.37x of their published numbers; prefill cells did reproduce.
