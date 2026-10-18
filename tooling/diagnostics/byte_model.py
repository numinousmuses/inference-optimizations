#!/usr/bin/env python3
"""Byte-level checkpoint model: tensor bytes by group and dtype, read straight from the safetensors headers on HF.

  python3 operations/kernels/byte_model.py <repo> <revision> [--out file.json]

Only the 8-byte length + JSON header of each shard is fetched (HTTP range), so a 1.6 TB checkpoint costs a few MB.
Groups: embed, lm_head, mtp, vision, routed experts, shared experts, attention (q/kv/o, MLA latent, indexer),
linear attention (GDN/KDA), dense mlp, gate/router, norms/other. Output: bytes per group and dtype, plus the config.
"""
import argparse, collections, concurrent.futures as cf, json, os, re, struct, urllib.request

HF = "https://huggingface.co"


def get(url, rng=None, timeout=60):
    h = {"Range": rng} if rng else {}
    tok = os.environ.get("HF_TOKEN") or (open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
                                         if os.path.exists(os.path.expanduser("~/.cache/huggingface/token")) else None)
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=timeout).read()


def header(repo, rev, f):
    u = f"{HF}/{repo}/resolve/{rev}/{f}"
    n = struct.unpack("<Q", get(u, "bytes=0-7"))[0]
    return json.loads(get(u, f"bytes=8-{7 + n}"))


RULES = [
    ("vision", r"visual|vision"),
    ("mtp", r"^mtp|\.mtp\.|nextn|layers\.(\d+)\.eh_proj|shared_head"),
    ("embed", r"embed_tokens|wte|tok_embeddings|\bembed\b"),
    ("lm_head", r"lm_head|output\.weight$|^head\."),
    ("routed_experts", r"\.experts\.|block_sparse_moe\.experts|ffn\.experts"),
    ("shared_experts", r"shared_expert"),
    ("router", r"\.gate\.(weight|e_score)|router|\.gate\.bias|e_score_correction"),
    ("indexer", r"indexer"),
    ("linear_attn", r"linear_attn|\.kda\.|gated_delta|\.ssm\."),
    ("attention", r"self_attn|\.attn\.|attention"),
    ("dense_mlp", r"\.mlp\.|feed_forward|\.ffn\."),
    ("norm", r"norm|ln_|layernorm"),
]


def group(name):
    for g, pat in RULES:
        if re.search(pat, name):
            return g
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo"); ap.add_argument("revision"); ap.add_argument("--out")
    a = ap.parse_args()
    cfg = json.loads(get(f"{HF}/{a.repo}/resolve/{a.revision}/config.json"))
    try:
        idx = json.loads(get(f"{HF}/{a.repo}/resolve/{a.revision}/model.safetensors.index.json"))["weight_map"]
        files = sorted(set(idx.values()))
    except Exception:  # noqa: BLE001 single-file checkpoint
        files = ["model.safetensors"]
    by = collections.Counter(); bydt = collections.Counter(); layer_max = 0
    with cf.ThreadPoolExecutor(24) as ex:
        for h in ex.map(lambda f: header(a.repo, a.revision, f), files):
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                s, e = v["data_offsets"]
                g = group(k)
                by[g] += e - s; bydt[(g, v["dtype"])] += e - s
                m = re.search(r"layers\.(\d+)\.", k)
                if m:
                    layer_max = max(layer_max, int(m.group(1)))
    res = {"repo": a.repo, "revision": a.revision, "shards": len(files), "total_bytes": sum(by.values()),
           "groups": dict(by), "groups_dtype": {f"{g}|{d}": b for (g, d), b in bydt.items()},
           "layers_seen": layer_max + 1, "config": cfg}
    print(f"{a.repo}@{a.revision[:8]} shards {len(files)} total {sum(by.values()) / 1e9:.2f} GB")
    for (g, d), b in sorted(bydt.items(), key=lambda x: -x[1]):
        print(f"  {g:16} {d:9} {b / 1e9:8.2f} GB")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
