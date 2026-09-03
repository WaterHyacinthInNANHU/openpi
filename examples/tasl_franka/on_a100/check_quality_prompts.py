"""CPU-only check that the CFG finetune config really tags its prompts.

Builds the dataset exactly the way the JAX training loader does for a stage-2 quality
config (prompt_from_task -> PresentationKeyedDataset -> repack group with the tag transform
at its head) and reports the realized tagged fraction plus a few example prompts.

    .venv/bin/python examples/tasl_franka/on_a100/check_quality_prompts.py \
        --config pi05_cotrain_franka_lora_10task_cfg_v2 --n 2000
"""
import argparse
import collections

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import quality_conditioning as _qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_cotrain_franka_lora_10task_cfg_v2")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    cfg = _config.get_config(args.config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    cond = _qc.stage2_conditioning(data_config)
    print("stage2_conditioning:", cond)
    dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    print("dataset rows after idle filter:", len(dataset))
    if cond is not None:
        dataset, sampler = _qc.wrap_presentations(dataset, cond, seed=cfg.seed, shuffle=True, rows=None)
        indices = iter(sampler)
    else:
        indices = iter(range(len(dataset)))
    repacked = _data_loader.TransformedDataset(dataset, list(data_config.repack_transforms.inputs))

    tagged = 0
    examples = collections.OrderedDict()
    for k, idx in zip(range(args.n), indices):
        prompt = str(repacked[idx]["prompt"])
        t = _qc.is_tagged(prompt)
        tagged += int(t)
        base = prompt.split(_qc.PROMPT_TAG_MARKER)[0]
        examples.setdefault(base, [0, 0])[int(t)] += 1
    print(f"sampled {args.n}: tagged {tagged} ({tagged / args.n:.4f}); expected ~0.8075 for a tagged config, 0 otherwise")
    for base, (untagged, tg) in examples.items():
        print(f"  tagged={tg:4d} untagged={untagged:4d}  {base!r}")
    if cond is not None:
        print("example tagged prompt:", repr(next(str(repacked[i]["prompt"]) for i in iter(sampler) if _qc.is_tagged(repacked[i]["prompt"]))))


if __name__ == "__main__":
    main()
