"""Exercise 3 — the AWQ scales ride inside the shards, and the files a weights backup drops let the replica start wrong.

    Design a DR manifest for a 70B AWQ-quantized model served in vLLM with 5
    LoRA adapters. List every file and config.

Reading of the exercise: "every file" is taken literally, so the model part
comes from a real 70B AWQ repository rather than from memory:
`hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`, whose file list,
`config.json`, `generation_config.json` and `model.safetensors.index.json`
were fetched on 2026-09-26. The vLLM defaults come from `vllm/config/lora.py`,
`vllm/config/model.py` and `vllm/sampling_params.py` on `main`, fetched the
same day. The 5 adapters and their ranks are an assumed deployment. The
manifest is data. `restore()` replays a backup against it and sorts each
missing entry by what it breaks.

**ANSWER: 47 entries in four groups, and a weights-only backup restores 10 of
them.** The groups are the model repository (all 19 files), the 5 adapters
(`adapter_config.json` and `adapter_model.safetensors` each, plus a registry
of name, path, base model and rank), the engine config (8 settings, among
them `--max-loras 5` and `--max-lora-rank 64`), and the deployment (9: image
digest, lockfile, Dockerfile, K8s YAML, autoscaler and router config, secret
references, checksums, runbook). A backup of `*.safetensors` plus the index
restores the 9 shards and the index. Of the 37 entries it misses, 3 stop the
replica starting (`config.json` and the two tokenizer files, the second of
which carries the chat template), 10 take adapters down, 10 change what a
started replica serves, and 14 are needed to rebuild the stack or keep an
exact, licensed copy.

**FINDING: the AWQ scales and zero points cannot be forgotten separately.**
The shard index maps 560 `qweight`, 560 `scales` and 560 `qzeros` tensors
into the same 9 shards, 39.77 GB in all, and the quantization settings (4
bits, group size 128, GEMM, zero point) are a block inside `config.json`.
The repository has no `quantize_config.json`. What a weights backup drops is
the small JSON next to the shards.

**FINDING: a restore that starts can still be wrong.** Without
`generation_config.json`, vLLM's `generation_config="auto"` finds nothing and
sampling falls to its defaults of temperature 1.0 and top_p 1.0, instead of
the repository's 0.6 and 0.9. Without the engine config, `max_loras`
defaults to 1, so the 5 adapters share one GPU slot. `max_lora_rank`
defaults to 16, so the adapters at rank 32 and 64 do not load. A drill that
only checks that the replica comes up passes all three.

**FINDING: the lesson's "three-file minimum" is three categories.** The
same manifest spans 47 entries. The lesson's "32%" figure was not found in
the two further-reading sources checked (TianPan, and BentoML, now the
Modular handbook). It is left unverified.

Structure: `REPO`, `VLLM` and `ADAPTERS` hold the sourced and assumed facts.
`manifest()` builds (path, group, consequence) rows, and `restore()` diffs a
backup against them.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "11-multi-region-kv-locality"
REPO = {  # huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4, 2026-09-26
    "shards": [f"model-{i:05d}-of-00009.safetensors" for i in range(1, 10)],
    "start": ("config.json", "model.safetensors.index.json", "tokenizer.json",
              "tokenizer_config.json"),  # tokenizer_config.json carries the chat template
    "behaviour": ("generation_config.json",),
    "provenance": ("special_tokens_map.json", ".gitattributes", "LICENSE", "README.md",
                   "USE_POLICY.md"),  # exact copy and licence terms; not tested for startup
    "quant": {"bits": 4, "group_size": 128, "quant_method": "awq", "version": "gemm",
              "zero_point": True},
    "tensors": {"qweight": 560, "scales": 560, "qzeros": 560}, "shard_gb": 39.77,
    "sampling": {"temperature": 0.6, "top_p": 0.9},
}
VLLM = {"max_loras": 1, "max_lora_rank": 16, "sampling": {"temperature": 1.0, "top_p": 1.0}}
ADAPTERS = {"fraud": 8, "kyc": 16, "support": 16, "collections": 32, "compliance": 64}
ENGINE = ("--quantization awq_marlin", "--enable-lora", "--max-loras 5",
          f"--max-lora-rank {max(ADAPTERS.values())}", "--max-cpu-loras 5",
          "--lora-modules (5 name=path pairs)", "--tensor-parallel-size / --max-model-len",
          "--gpu-memory-utilization / --served-model-name")
DEPLOY = ("image digest (vllm version pinned)", "dependency lockfile", "Dockerfile",
          "k8s Deployment + Service YAML", "autoscaler config", "router config (prefix hash)",
          "secret references (not values)", "sha256 of every file above", "restore runbook")


def manifest():
    rows = [(f, "model", "start") for f in REPO["shards"] + list(REPO["start"])]
    rows += [(f, "model", "behaviour") for f in REPO["behaviour"]]
    rows += [(f, "model", "provenance") for f in REPO["provenance"]]
    for name in ADAPTERS:
        rows += [(f"adapters/{name}/{f}", "adapter", "adapter")
                 for f in ("adapter_config.json", "adapter_model.safetensors")]
    rows.append(("adapters/registry (name -> path, base model, rank)", "adapter", "behaviour"))
    rows += [(flag, "engine", "behaviour") for flag in ENGINE]
    rows += [(item, "deploy", "rebuild") for item in DEPLOY]
    return rows


def restore(backup):
    """Missing manifest entries by consequence."""
    missing = [r for r in manifest() if r[0] not in backup]
    return dict(collections.Counter(r[2] for r in missing))


def default_restore():
    """What a replica restored without the engine config serves."""
    loaded = [n for n, rank in ADAPTERS.items() if rank <= VLLM["max_lora_rank"]]
    return {"slots": VLLM["max_loras"], "loaded": loaded, "sampling": VLLM["sampling"]}


def solve():
    rows = manifest()
    weights = set(REPO["shards"]) | {"model.safetensors.index.json"}
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "total": len(rows), "groups": dict(collections.Counter(r[1] for r in rows)),
        "weights_hit": sum(r[0] in weights for r in rows), "weights_miss": restore(weights),
        "files": sorted(REPO["shards"] + list(REPO["start"] + REPO["behaviour"]
                                              + REPO["provenance"])),
        "defaults": default_restore(),
        "doc": ("three-file minimum" in doc, "AWQ scales" in doc, "32%" in doc),
    }


def verify(result):
    d, miss = result["defaults"], result["weights_miss"]
    return [
        practice.Check(
            "ANSWER: 47 entries in four groups, and a weights-only backup restores 10",
            result["total"] == 47 and result["weights_hit"] == 10
            and miss == {"start": 3, "adapter": 10, "behaviour": 10, "provenance": 5,
                         "rebuild": 9},
            f"groups {result['groups']}; a *.safetensors + index backup restores "
            f"{result['weights_hit']} and misses {miss}",
        ),
        practice.Check(
            "FINDING: the AWQ scales and zero points cannot be forgotten separately",
            len(result["files"]) == 19 and "quantize_config.json" not in result["files"]
            and REPO["tensors"]["scales"] == REPO["tensors"]["qzeros"] == 560,
            f"{len(result['files'])} files in the repo, no quantize_config.json; "
            f"{REPO['tensors']} tensors live in 9 shards ({REPO['shard_gb']} GB); "
            f"quantization_config {REPO['quant']} is a block of config.json",
        ),
        practice.Check(
            "FINDING: a restore that starts can still be wrong",
            d["slots"] == 1 and d["loaded"] == ["fraud", "kyc", "support"]
            and d["sampling"] != REPO["sampling"],
            f"without generation_config.json sampling is {d['sampling']} not "
            f"{REPO['sampling']}; default engine args give {d['slots']} LoRA slot and load "
            f"{d['loaded']} of {list(ADAPTERS)}",
        ),
        practice.Check(
            "FINDING: the lesson's 'three-file minimum' is three categories",
            all(result["doc"]) and result["total"] > 3 * 10,
            f"the lesson says 'three-file minimum' and names 'AWQ scales' as files to back "
            f"up; the manifest is {result['total']} entries",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
