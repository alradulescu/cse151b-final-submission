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
base_max_tokens: 16384
base_max_model_len: 32768
prompt_style: cot
temperature: 0.6
seed: 42
model_postprocess: false
```

Because this is a sampled prompting setup, exact outputs are not guaranteed to
be string-identical across hardware, CUDA drivers, or vLLM versions. Accuracy
can vary by a few percentage points from run to run; the pinned settings above
are the configuration used for the submitted candidate.

The code uses local model inference and standard CSV/file handling. It does not
make external model calls or use tool-augmented generation at inference time.

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

## Secondary Tuned Route

The repository also keeps a secondary tuned route. It is not the default
pipeline. To run it, set the adapter locations and call
`pipeline="legal_hybrid_1027"`.

Adapter variables:

```bash
export CSE151B_SELECTOR_ALL_REPAIR_MODEL=alradulescu/cse151b-repair_trace_lora_all_e3_r64
export CSE151B_SELECTOR_FREEFORM_REPAIR_MODEL=alradulescu/cse151b-repair_trace_lora_freeform_e3_r64
export CSE151B_MCQ_REPAIR_MODEL=alradulescu/cse151b-repair_trace_lora_mcq_e5_lr5e5_r64
export CSE151B_FREEFORM_STRUCTURED_MODEL=alradulescu/cse151b-repair_trace_lora_freeform_structured_e8_lr5e5_r64
```

Pass names:

```text
selector_all_repair
selector_freeform_repair
mcq_repair
freeform_structured
```

Example:

```python
from run_inference import run_inference

run_inference(
    data_path="kaggle_data/private.jsonl",
    output_csv="submission_tuned.csv",
    work_dir="results/tuned_route",
    pipeline="legal_hybrid_1027",
    hybrid_final_policy="full",
)
```

For the lighter selector-only variant, use `hybrid_final_policy="selector"`.
