# Team G092 Goats Final Submission

This repository provides Team G092 Goats' final inference entry point for
generating a Kaggle submission CSV with the required model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Hardware and time estimate: the submitted pipeline is intended for AWS
`g6e.12xlarge` with four NVIDIA L40S GPUs. Full private-set inference is
expected to take about `35-45 minutes` on four L40S cards, or about
`2h10m-2h30m` on one L40S card.

The default route is a raw Qwen generation configuration with a 16k-token
generation budget. In this repository, `16k` means
`base_max_tokens=16384`; it is not a separate model name.

## Single Entry Point

Use `run_inference()` from `run_inference.py`:

```python
from run_inference import run_inference

run_inference(
    data_path="kaggle_data/private.jsonl",
    output_csv="submission.csv",
    work_dir="results/final_single_entry",
)
```

The private dataset is not included in this repository. To run on a dataset,
place the JSONL at `kaggle_data/private.jsonl` or pass its path as `data_path`.

Equivalent command-line handoff:

```bash
python3 scripts/run_raw_qwen16k_primary.py \
  --data kaggle_data/private.jsonl \
  --output submission.csv \
  --work-dir results/raw_qwen16k_primary
```

The wrapper script calls the same default `run_inference()` path.

## Model Weights

The default path downloads the required base model from Hugging Face:

```text
Qwen/Qwen3-4B-Thinking-2507
```

No adapter weights are required for the default submission path.

## Pipeline

`run_inference()` defaults to `pipeline="single_model"` and
`model_postprocess=False`. It performs:

1. load `Qwen/Qwen3-4B-Thinking-2507`;
2. generate one response for every input row using the pinned settings below;
3. write the final `id,response` CSV in input row order.

Pinned default settings:

```text
backend: vLLM
vllm_enforce_eager: true
vllm_batch_size: 1024
base_max_tokens: 16384
base_max_model_len: 32768
prompt_style: cot
temperature: 0.7
seed: 42
model_postprocess: false
```

Because this is a sampled prompting setup, exact outputs are not guaranteed to
be string-identical across hardware, CUDA drivers, or vLLM versions. Accuracy
can vary by a few percentage points from run to run; the pinned settings above
are the configuration used for the submitted candidate.

The code uses local model inference and standard CSV/file handling. It does not
make external model calls or use tool-augmented generation at inference time.

## Experimental Multipass Route

This branch also includes a single-entry multipass route for testing a fixed
row-type strategy. This route does not require or load fine-tuned adapters. It
uses only the required base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

The route is exposed as `pipeline="base_multipass_route"` and performs all
stages inside one `run_inference()` call:

1. split the provided input rows by schema into multiple-choice and free-form
   subsets;
2. regenerate the multiple-choice path with compact boxed prompting followed by
   a Qwen structured boxed pass;
3. regenerate the free-form path with 16k CoT prompting followed by a Qwen solve
   boxed pass;
4. merge the two generated outputs by fixed row type and write the final CSV.

It generates every answer-changing stage from the provided input rows. The
non-model code only writes temporary subset JSONL files, merges fixed row types,
does response-string trimming/LaTeX wrapper cleanup, and writes CSV/JSONL files.

Run it with:

```bash
python3 scripts/run_base_multipass_route.py \
  --data kaggle_data/private.jsonl \
  --output submission.csv \
  --work-dir results/base_multipass_route
```

Equivalent Python call:

```python
from run_inference import run_inference

run_inference(
    data_path="kaggle_data/private.jsonl",
    output_csv="submission.csv",
    work_dir="results/base_multipass_route",
    pipeline="base_multipass_route",
)
```

## Environment Setup

Use a CUDA-capable Linux GPU environment. The final generation runs used an AWS
`g6e.12xlarge` instance with four NVIDIA L40S GPUs. For the 943-row private
set, expected total inference time is about `35-45 minutes` when sharded across
four L40S GPUs, or about `2h10m-2h30m` on one L40S. Runtime varies with
sequence lengths, driver versions, and vLLM version.

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install vllm peft bitsandbytes huggingface_hub
```

Then run:

```bash
python3 - <<'PY'
from run_inference import run_inference

run_inference(
    data_path="kaggle_data/private.jsonl",
    output_csv="submission.csv",
    work_dir="results/final_single_entry",
)
PY
```
