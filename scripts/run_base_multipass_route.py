#!/usr/bin/env python3
"""Run the base-Qwen fixed row-type multipass route from one entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_inference import run_inference  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "base_multipass_route.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="kaggle_data/private.jsonl")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--work-dir", default="results/base_multipass_route")
    parser.add_argument("--gpu-id")
    parser.add_argument("--backend", choices=["vllm", "transformers", "auto"])
    parser.add_argument("--vllm-batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    summary = run_inference(
        data_path=args.data,
        output_csv=args.output,
        work_dir=args.work_dir,
        model=cfg["model"],
        pipeline=cfg["pipeline"],
        backend=args.backend or cfg["backend"],
        gpu_id=args.gpu_id,
        vllm_batch_size=int(args.vllm_batch_size or cfg["vllm_batch_size"]),
        vllm_enforce_eager=bool(cfg["vllm_enforce_eager"]),
        temperature=float(cfg["temperature"]),
        seed=int(args.seed if args.seed is not None else cfg["seed"]),
        self_consistency_n=int(cfg["self_consistency_n"]),
        model_postprocess=bool(cfg["model_postprocess"]),
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps({"config": str(CONFIG_PATH), "summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
