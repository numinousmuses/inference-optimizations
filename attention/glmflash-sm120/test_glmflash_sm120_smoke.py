"""CPU test of the smoke script's config trim against the real RadixArk/GLM-5.3-Flash-NVFP4 config (HF cache)."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glmflash_sm120_smoke as S  # noqa: E402


def _cfg():
    try:
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(S.REPO_ID, "config.json")))
    except Exception:
        pytest.skip("HF config not reachable")


def test_trim_keeps_every_dsa_dim():
    c = _cfg()
    t = S.trim_config(c)
    tt, ot = t["text_config"], c["text_config"]
    assert "quantization_config" not in t and tt["num_hidden_layers"] == 8
    assert tt["layer_types"] == ["linear_attention"] * 3 + ["deepseek_sparse_attention"] + ["linear_attention"] * 3 \
        + ["deepseek_sparse_attention"]
    assert tt["linear_attn_config"]["kda_layers"] == [0, 1, 2, 4, 5, 6]
    for k in ("kv_lora_rank", "qk_rope_head_dim", "num_attention_heads", "index_topk", "index_kpool", "index_n_heads",
              "index_head_dim", "q_lora_rank", "qk_nope_head_dim", "v_head_dim", "num_nextn_predict_layers",
              "mla_use_nope", "hidden_size"):
        assert tt[k] == ot[k], k
    assert tt["qk_rope_head_dim"] == 0 and tt["index_topk"] == 2048 and tt["index_kpool"] == 4
    assert tt["n_routed_experts"] == 16 and len(tt["mlp_layer_types"]) == 8
