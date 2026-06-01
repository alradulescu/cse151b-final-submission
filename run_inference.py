"""
run_inference.py
Run Qwen3-4B-Thinking on all questions using vLLM or Transformers.

Usage:
  # Public dataset (with scoring):
  python run_inference.py --data kaggle_data/private.jsonl --output results/private_results.jsonl

  # Private test set (for Kaggle submission):
  python run_inference.py --data data/private.jsonl --output results/private_results.jsonl --no-score

  # Resume an interrupted run:
  python run_inference.py --data kaggle_data/private.jsonl --output results/private_results.jsonl --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step, showing your reasoning. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

# ── LoRA adapter detection ────────────────────────────────────────────────────

def is_lora_adapter(model_path: str) -> bool:
    """Return True if model_path is a PEFT/LoRA adapter directory."""
    return (Path(model_path) / "adapter_config.json").exists()


def load_model_and_tokenizer(model_path: str, bnb_config):
    """Load a base model directly, or load and merge a PEFT/LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if is_lora_adapter(model_path):
        adapter_config = json.loads((Path(model_path) / "adapter_config.json").read_text())
        base_model_name = adapter_config.get(
            "base_model_name_or_path",
            "Qwen/Qwen3-4B-Thinking-2507",
        )
        print(f"[LoRA] Detected adapter at {model_path}")
        print(f"[LoRA] Loading base model: {base_model_name}")

        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
        )
        base_model.config.use_cache = True
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("[LoRA] Adapter merged into base model.")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        print(f"Loading base model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
        )
        model.config.use_cache = True
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


SYSTEM_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices carefully, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

SYSTEM_MATH_DIRECT = (
    "You are an expert mathematician. Solve the problem, but keep the response concise. "
    "Output the final answer inside \\boxed{}. Do not include long reasoning."
)

SYSTEM_MCQ_DIRECT = (
    "You are an expert mathematician. Read the problem and answer choices, then output "
    "ONLY the letter of the single best option inside \\boxed{}, e.g. \\boxed{C}."
)

SYSTEM_MATH_STRICT = (
    "You are an expert mathematician. Solve the problem carefully. "
    "Every [ANS] blank, part, or requested sub-answer must be answered. "
    "Put the final answer at the very end inside exactly one \\boxed{}. "
    "If there are multiple answers, put all of them in the original order inside "
    "that same \\boxed{}, separated by commas. Do not leave an [ANS] blank unresolved."
)

SYSTEM_MCQ_STRICT = (
    "You are an expert mathematician. Read the problem and answer choices carefully. "
    "Choose exactly one listed option. Put the final option letter at the very end "
    "inside exactly one \\boxed{}, e.g. \\boxed{C}. Do not output an unlisted letter."
)

SYSTEM_MATH_COMPACT_BOXED = (
    "You are a careful mathematician. Solve briefly and directly. "
    "Do not restate the problem. Do not show exploratory reasoning or long derivations. "
    "Compute the answer, then output exactly one final answer inside \\boxed{} and stop. "
    "Nothing may appear after the boxed answer."
)

SYSTEM_MCQ_COMPACT_BOXED = (
    "You are solving a multiple choice math problem. Determine the single best option. "
    "Output exactly one uppercase letter inside \\boxed{} and stop. "
    "Do not include option text, punctuation, or explanation after the box."
)

SYSTEM_MATH_FINAL_BOXED = (
    "Answer the math problem with the shortest correct response possible. "
    "Preserve exact symbolic expressions when possible. "
    "Always output the final answer inside exactly one \\boxed{} and stop."
)

SYSTEM_MCQ_FINAL_BOXED = (
    "Answer the multiple choice problem with only the chosen option letter. "
    "Always output exactly one uppercase letter inside exactly one \\boxed{} and stop."
)

# Few-shot demonstrations. Kept short so they don't blow the context budget when
# combined with thinking-mode generations. Drawn from the public set's question
# styles (numerical free-form, expression free-form, and an MCQ).
FEWSHOT_FREEFORM = [
    {
        "q": "Find the sum of the first 100 positive even whole numbers.",
        "a": (
            "The first 100 positive even numbers are 2, 4, 6, ..., 200. "
            "Their sum is 2 * (1 + 2 + ... + 100) = 2 * (100 * 101 / 2) = 10100. "
            "\\boxed{10100}"
        ),
    },
    {
        "q": "Simplify: $\\int_{-\\infty}^{\\infty} e^{-x^2} dx$.",
        "a": (
            "This is the Gaussian integral, equal to \\sqrt{\\pi}. "
            "\\boxed{\\sqrt{\\pi}}"
        ),
    },
]

FEWSHOT_MCQ = [
    {
        "q": "What is the value of $2^{10}$?",
        "opts": ["512", "1000", "1024", "2048", "4096"],
        "a": (
            "2^{10} = 2*2*2*2*2*2*2*2*2*2 = 1024. That matches option C. \\boxed{C}"
        ),
    },
]


def _format_mcq_user(question: str, options: list) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
    return f"{question}\n\nOptions:\n{opts_text}"


def build_messages(question: str, options: Optional[list], prompt_style: str = "cot") -> list:
    """Returns chat messages list for tokenizer.apply_chat_template."""
    if options:
        if prompt_style == "direct":
            system = SYSTEM_MCQ_DIRECT
        elif prompt_style == "strict":
            system = SYSTEM_MCQ_STRICT
        elif prompt_style == "compact_boxed":
            system = SYSTEM_MCQ_COMPACT_BOXED
        elif prompt_style == "final_boxed":
            system = SYSTEM_MCQ_FINAL_BOXED
        else:
            system = SYSTEM_MCQ
        user = _format_mcq_user(question, options)
    else:
        if prompt_style == "direct":
            system = SYSTEM_MATH_DIRECT
        elif prompt_style == "strict":
            system = SYSTEM_MATH_STRICT
        elif prompt_style == "compact_boxed":
            system = SYSTEM_MATH_COMPACT_BOXED
        elif prompt_style == "final_boxed":
            system = SYSTEM_MATH_FINAL_BOXED
        else:
            system = SYSTEM_MATH
        user = question

    messages = [{"role": "system", "content": system}]

    if prompt_style == "fewshot":
        examples = FEWSHOT_MCQ if options else FEWSHOT_FREEFORM
        for ex in examples:
            ex_user = _format_mcq_user(ex["q"], ex["opts"]) if options else ex["q"]
            messages.append({"role": "user",      "content": ex_user})
            messages.append({"role": "assistant", "content": ex["a"]})

    messages.append({"role": "user", "content": user})
    return messages


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Legacy 2-tuple interface kept for callers that still use it."""
    if options:
        return SYSTEM_MCQ, _format_mcq_user(question, options)
    return SYSTEM_MATH, question


def render_chat_prompt(tokenizer, messages: list, args) -> str:
    """Render chat prompts, disabling Qwen thinking mode when requested."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)

    # Qwen3-Thinking-2507 still renders an open `<think>` block even when the
    # tokenizer accepts enable_thinking=False. Close it explicitly so concise
    # answer-only smoke tests do not spend the entire budget on hidden reasoning.
    if args.disable_thinking and prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n\n</think>\n\n"
    return prompt


# ── Scoring helpers ───────────────────────────────────────────────────────────

def extract_letter(text: str, options: Optional[list] = None) -> str:
    """Best-effort MCQ letter extraction.

    Many responses get truncated mid-reasoning at max_tokens, so we look for
    several patterns in priority order:
      1. \\boxed{X}             — clean case
      2. "answer is X" / "answer: X" / "(X)" / etc.
      3. Last standalone capital letter that's a valid option
    """
    if not text:
        return ""

    # 1. Clean boxed answer
    m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if m:
        return m.group(1).upper()

    # Build the legal letter set if options were provided
    if options:
        legal = {chr(65 + i) for i in range(len(options))}
    else:
        legal = set("ABCDEFGHIJ")

    # 2. Phrase patterns — search the LAST 1500 chars (where the conclusion
    # usually lives), case-insensitively. Patterns are ordered by reliability.
    tail = text[-1500:].upper()
    patterns = [
        r"FINAL ANSWER\s*[:\-]?\s*\(?([A-J])\b",
        r"CORRECT ANSWER IS\s*\(?([A-J])\b",
        r"ANSWER IS\s*\(?([A-J])\b",
        r"ANSWER\s*[:\-]\s*\(?([A-J])\b",
        r"OPTION\s*\(?([A-J])\)?\s*IS\s*(?:THE\s+)?(?:CORRECT|RIGHT)",
        r"CHOOSE\s+(?:OPTION\s+)?\(?([A-J])\b",
        r"SELECT\s+(?:OPTION\s+)?\(?([A-J])\b",
        r"\bTHE\s+ANSWER\s+(?:IS|MUST\s+BE)\s+\(?([A-J])\b",
        r"\(([A-J])\)\s+IS\s+(?:THE\s+)?CORRECT",
    ]
    for pat in patterns:
        hits = re.findall(pat, tail)
        for h in reversed(hits):
            if h in legal:
                return h

    # 3. Last standalone single capital letter that's a valid option
    matches = re.findall(r"\b([A-J])\b", text)
    for h in reversed(matches):
        if h in legal:
            return h

    # Last resort: still require the final capital-letter guess to be legal.
    upper_matches = re.findall(r"\b([A-Z])\b", text.upper())
    for h in reversed(upper_matches):
        if h in legal:
            return h
    return ""


def _extract_boxed(text: str) -> str:
    """Extracts the *last* \\boxed{...} content with brace matching."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return ""
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip()


def majority_vote(samples: list, is_mcq: bool) -> str:
    """Pick a single response from N samples via majority vote on extracted answer.

    Returns the FULL response text of one of the samples whose extracted answer
    matched the mode — that way downstream scoring (which re-extracts) sees a
    consistent choice and the response includes intact reasoning.
    """
    from collections import Counter
    if not samples:
        return ""
    if len(samples) == 1:
        return samples[0]
    if is_mcq:
        keys = [extract_letter(s) for s in samples]
    else:
        # Normalise extracted answer: strip whitespace, drop \! / \, spacing latex
        keys = []
        for s in samples:
            k = _extract_boxed(s)
            k = re.sub(r"\s+", "", k)
            k = k.replace("\\,", "").replace("\\!", "").replace("\\;", "")
            keys.append(k)
    counts = Counter(k for k in keys if k)
    if not counts:
        return samples[0]
    winner_key, _ = counts.most_common(1)[0]
    for s, k in zip(samples, keys):
        if k == winner_key:
            return s
    return samples[0]


def score_item(item: dict, response: str, judger) -> bool:
    gold = item["answer"]
    if item.get("options"):
        return extract_letter(response, item.get("options")) == str(gold).strip().upper()

    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        # options should be a list of lists, one for each gold answer
        options_list = [item.get("options", [])] * len(gold_list)
        return judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=options_list,
        )[0]
    except Exception:
        return False


def make_record(item: dict, response: str, judger) -> dict:
    record = {
        "id":     item["id"],
        "is_mcq": bool(item.get("options")),
        "response": response,
    }
    if judger is not None:
        record["gold"] = item["answer"]
        record["correct"] = score_item(item, response, judger)
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp_path.replace(path)


def print_accuracy_summary(records: list[dict]) -> None:
    mcq_res  = [r for r in records if r.get("is_mcq") and "correct" in r]
    free_res = [r for r in records if not r.get("is_mcq") and "correct" in r]
    if not mcq_res and not free_res:
        return

    def acc(s): return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    all_scored = mcq_res + free_res
    print(f"  Overall    : {sum(r['correct'] for r in all_scored):4d} / {len(all_scored):4d}  ({acc(all_scored):.2f}%)")
    print("=" * 50)


# ── vLLM inference ────────────────────────────────────────────────────────────

def run_vllm(data, tokenizer, args, on_batch=None):
    from vllm import LLM, SamplingParams

    print("Loading model with vLLM...")
    llm_kwargs = dict(
        model=args.model,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=args.vllm_max_num_seqs,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        dtype=args.dtype,
        enforce_eager=getattr(args, "vllm_enforce_eager", False),
    )
    if args.vllm_quantization != "none":
        llm_kwargs["quantization"] = args.vllm_quantization
        if args.vllm_quantization == "bitsandbytes":
            llm_kwargs["load_format"] = "bitsandbytes"

    llm = LLM(**llm_kwargs)
    n = max(1, args.self_consistency_n)
    sampling_params = SamplingParams(
        n=n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        seed=getattr(args, "seed", 42),
    )
    print(
        "Model loaded. "
        f"prompt_style={args.prompt_style} n={n} "
        f"quantization={args.vllm_quantization} dtype={args.dtype}"
    )

    # Build all prompts
    prompts = []
    for item in data:
        messages = build_messages(item["question"], item.get("options"), args.prompt_style)
        prompt_text = render_chat_prompt(tokenizer, messages, args)
        prompts.append(prompt_text)

    chunk_size = args.vllm_batch_size
    print(f"Generating {len(prompts)} prompts x {n} samples with vLLM, chunk_size={chunk_size}...")
    responses = []
    for batch_start in tqdm(range(0, len(prompts), chunk_size), desc="Generating"):
        batch_items = data[batch_start : batch_start + chunk_size]
        batch_prompts = prompts[batch_start : batch_start + chunk_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        batch_responses = []
        for item, out in zip(batch_items, outputs):
            samples = [c.text.strip() for c in out.outputs]
            if n == 1:
                batch_responses.append(samples[0])
            else:
                batch_responses.append(majority_vote(samples, is_mcq=bool(item.get("options"))))

        responses.extend(batch_responses)
        if on_batch is not None:
            on_batch(batch_items, batch_responses)

    return responses


# ── Transformers inference ────────────────────────────────────────────────────

def run_transformers(data, tokenizer_arg, args, on_batch=None):
    """Run inference with the Transformers backend.

    Handles both plain base models and LoRA adapter directories.
    `tokenizer_arg` is ignored when model_path is an adapter (we reload
    from the adapter dir to get the right special tokens); it is used
    otherwise so we don't double-load.
    """
    import torch
    import time
    from transformers import set_seed
    from transformers import BitsAndBytesConfig

    start_time = time.time()
    set_seed(getattr(args, "seed", 42))
    print("Loading model with Transformers (INT4 QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    llm, tokenizer = load_model_and_tokenizer(args.model, bnb_config)
    print("Model loaded.")

    responses = []
    batch_size = args.batch_size
    total_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, len(data), batch_size)):
        batch = data[batch_start : batch_start + batch_size]
        prompts = []
        for item in batch:
            messages = build_messages(item["question"], item.get("options"), args.prompt_style)
            prompt_text = render_chat_prompt(tokenizer, messages, args)
            prompts.append(prompt_text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len,
        ).to(llm.device)

        with torch.no_grad():
            do_sample = args.temperature > 0
            generation_kwargs = dict(
                max_new_tokens=args.max_tokens,
                top_p=0.95,
                top_k=20,
                do_sample=do_sample,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
            output_ids = llm.generate(
                **inputs,
                **generation_kwargs,
            )

        batch_responses = []
        for i, out in enumerate(output_ids):
            new_tokens = out[inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            batch_responses.append(decoded)

        responses.extend(batch_responses)

        # ── Progress print ────────────────────────────────────────────────
        done = min(batch_start + batch_size, len(data))
        pct = 100 * done / len(data)
        print(f"  [{done:4d}/{len(data)}  {pct:5.1f}%]  batch {batch_idx+1}/{total_batches}",
              flush=True)

        if on_batch is not None:
            on_batch(batch, batch_responses)
            
        if args.time_limit > 0 and (time.time() - start_time) / 60.0 > args.time_limit:
            print(f"\nTime limit of {args.time_limit} minutes reached. Stopping inference early!")
            break

    return responses


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_jsonl_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _strip_single_wrapping_quote(text: str) -> str:
    if len(text) < 2 or text[0] != text[-1] or text[0] not in {"'", '"'}:
        return text
    return text[1:-1].strip()


def _safe_final_string_cleanup(response: object) -> str:
    """Presentation-only cleanup; never inspect the problem or compute answers."""
    text = "" if response is None else str(response)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _strip_single_wrapping_quote(text)

    for left, right in (("$$", "$$"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) > 2 * len(left):
            text = text[len(left):-len(right)].strip()

    if text.startswith("boxed{"):
        text = "\\" + text

    boxed_parens = re.fullmatch(r"\\boxed\((.*)\)", text, flags=re.S)
    if boxed_parens:
        text = f"\\boxed{{{boxed_parens.group(1).strip()}}}"

    if text.startswith("\\boxed{") and text.count("{") == text.count("}") + 1:
        text = text + "}"

    boxed_braces = re.fullmatch(r"\\boxed\{(.*)\}", text, flags=re.S)
    if boxed_braces:
        text = f"\\boxed{{{boxed_braces.group(1).strip()}}}"

    return text


def _apply_safe_final_string_cleanup(records: list[dict]) -> tuple[list[dict], int]:
    cleaned_records: list[dict] = []
    changed_count = 0
    for record in records:
        cleaned = dict(record)
        original = cleaned.get("response", "")
        cleaned_response = _safe_final_string_cleanup(original)
        if cleaned_response != original:
            cleaned["response"] = cleaned_response
            cleaned["safe_cleanup"] = "trim_and_latex_wrapper_only"
            changed_count += 1
        cleaned_records.append(cleaned)
    return cleaned_records, changed_count


def _validate_competition_model(model: str) -> None:
    """Reject final-pipeline models outside the competition-designated family."""
    designated = "Qwen/Qwen3-4B-Thinking-2507"
    if model == designated:
        return

    model_path = Path(model)
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        try:
            config = json.loads(adapter_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid adapter_config.json for {model}: {exc}") from exc
        if config.get("base_model_name_or_path") == designated:
            return

    raise ValueError(
        "Final reproducible inference must use Qwen/Qwen3-4B-Thinking-2507 "
        "or a local LoRA adapter whose adapter_config.json names that base model. "
        f"Got: {model}"
    )


TUNED_HYBRID_PRESET = "tuned_hybrid"
RAW_DUAL_CHOOSE_PRESET = "raw_dual_choose"
DEFAULT_HYBRID_ADAPTERS = {
    "selector_all_repair": None,
    "selector_freeform_repair": None,
    "mcq_repair": None,
    "freeform_structured": None,
}


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_").lower()


def _resolve_final_model(model: str, cache_dir: Path) -> str:
    """Resolve a base model or adapter repo/path to a validated local model spec.

    Hugging Face LoRA adapters are downloaded into work_dir so qwen.py can
    inspect adapter_config.json and load the adapter without making any
    inference-time external model/API calls beyond ordinary weight download.
    """
    if model == "Qwen/Qwen3-4B-Thinking-2507":
        return model

    candidate = Path(model)
    if candidate.exists():
        _validate_competition_model(str(candidate))
        return str(candidate)
    if not candidate.is_absolute():
        repo_candidate = Path(__file__).resolve().parent / candidate
        if repo_candidate.exists():
            _validate_competition_model(str(repo_candidate))
            return str(repo_candidate)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError(
            f"Adapter {model!r} is not a local path and huggingface_hub is unavailable."
        ) from exc

    local_dir = cache_dir / _model_slug(model)
    snapshot_path = Path(
        snapshot_download(
            repo_id=model,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
    )
    _validate_competition_model(str(snapshot_path))
    return str(snapshot_path)


def _required_optional_model(value: str | None, env_name: str) -> str:
    if value:
        return value
    raise ValueError(
        f"{env_name} or the matching run_inference() argument is required "
        f"when pipeline={TUNED_HYBRID_PRESET!r} is selected."
    )


def _records_by_id(records: list[dict]) -> dict[int, dict]:
    return {int(record["id"]): record for record in records}


def _merge_row_type_outputs(
    items: list[dict],
    mcq_records: list[dict],
    freeform_records: list[dict],
    output_path: Path,
    stage_name: str,
) -> list[dict]:
    mcq_by_id = _records_by_id(mcq_records)
    freeform_by_id = _records_by_id(freeform_records)
    merged: list[dict] = []
    missing: list[int] = []

    for item in items:
        row_id = int(item["id"])
        source = mcq_by_id if bool(item.get("options")) else freeform_by_id
        if row_id not in source:
            missing.append(row_id)
            continue
        row = dict(source[row_id])
        row["single_entry_stage"] = stage_name
        row["row_type_route"] = "mcq" if bool(item.get("options")) else "freeform"
        merged.append(row)

    if missing:
        raise RuntimeError(
            f"{stage_name} did not produce rows for {len(missing)} ids; "
            f"first missing ids: {missing[:10]}"
        )
    _write_jsonl_records(output_path, merged)
    return merged


def _run_qwen_repair_stage(
    *,
    root: Path,
    data_path: Path,
    base_results: Path,
    output_path: Path,
    model: str,
    backend: str,
    question_filter: str,
    mode: str,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    base_tail_chars: int,
    gpu_id: str | None,
    resume: bool,
    reuse_existing: bool,
    boxed_prefill: bool,
) -> None:
    if reuse_existing and output_path.exists():
        print(f"Reusing {output_path}", flush=True)
        return

    cmd = [
        sys.executable,
        str(root / "qwen.py"),
        "--model",
        model,
        "--backend",
        backend,
        "--data",
        str(data_path),
        "--base-results",
        str(base_results),
        "--output",
        str(output_path),
        "--question-filter",
        question_filter,
        "--mode",
        mode,
        "--batch-size",
        str(batch_size),
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(max_model_len),
        "--base-tail-chars",
        str(base_tail_chars),
        "--disable-thinking",
        "--no-score",
    ]
    if gpu_id is not None:
        cmd.extend(["--gpu-id", str(gpu_id)])
    if resume:
        cmd.append("--resume")
    if boxed_prefill:
        cmd.append("--boxed-prefill")

    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write_submission_csv(output_csv: Path, items: list[dict], records: list[dict]) -> str:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    by_id = _records_by_id(records)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for item in items:
            row = by_id[int(item["id"])]
            writer.writerow([row["id"], row.get("response", "")])
    return hashlib.sha256(output_csv.read_bytes()).hexdigest()


def _run_tuned_hybrid_pipeline(
    *,
    data_path: Path,
    output_csv: Path,
    work_dir: Path,
    base_model: str,
    selector_all_repair_model: str,
    selector_freeform_repair_model: str,
    mcq_repair_model: str,
    freeform_structured_model: str,
    backend: str,
    gpu_id: str | None,
    base_max_tokens: int,
    base_max_model_len: int,
    prompt_style: str,
    temperature: float,
    seed: int,
    disable_thinking: bool,
    batch_size: int,
    vllm_batch_size: int,
    repair_backend: str,
    repair_batch_size: int,
    repair_max_tokens: int,
    repair_max_model_len: int,
    repair_base_tail_chars: int,
    resume: bool,
    reuse_existing: bool,
    boxed_prefill: bool,
    hybrid_final_policy: str,
) -> dict:
    """Rebuild the staged legal hybrid with one fixed, model-only pipeline."""
    root = Path(__file__).resolve().parent
    data_path = data_path if data_path.is_absolute() else root / data_path
    output_csv = output_csv if output_csv.is_absolute() else root / output_csv
    work_dir = work_dir if work_dir.is_absolute() else root / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    items = _load_jsonl_records(data_path)
    if hybrid_final_policy not in {"selector", "full"}:
        raise ValueError("hybrid_final_policy must be 'selector' or 'full'")

    model_cache = work_dir / "downloaded_adapters"
    resolved_selector_all = _resolve_final_model(selector_all_repair_model, model_cache)
    resolved_selector_free = _resolve_final_model(selector_freeform_repair_model, model_cache)
    resolved_mcq = (
        _resolve_final_model(mcq_repair_model, model_cache)
        if hybrid_final_policy == "full"
        else None
    )
    resolved_freeform = (
        _resolve_final_model(freeform_structured_model, model_cache)
        if hybrid_final_policy == "full"
        else None
    )
    _validate_competition_model(base_model)

    base_dir = work_dir / "00_base_qwen_seed"
    base_summary = run_inference(
        data_path=data_path,
        output_csv=base_dir / "base_seed.csv",
        work_dir=base_dir,
        model=base_model,
        backend=backend,
        gpu_id=gpu_id,
        base_max_tokens=base_max_tokens,
        base_max_model_len=base_max_model_len,
        prompt_style=prompt_style,
        temperature=temperature,
        seed=seed,
        disable_thinking=disable_thinking,
        batch_size=batch_size,
        vllm_batch_size=vllm_batch_size,
        self_consistency_n=1,
        model_postprocess=False,
        resume=resume,
        reuse_existing=reuse_existing,
        pipeline="single_model",
    )
    base_jsonl = Path(base_summary["results_jsonl"])

    qwen_backend = repair_backend
    if qwen_backend == "auto":
        qwen_backend = "vllm" if backend in {"auto", "vllm"} else "transformers"

    selector_mcq = work_dir / "01_selector_all_repair_mcq.jsonl"
    selector_free = work_dir / "02_selector_freeform_repair.jsonl"
    _run_qwen_repair_stage(
        root=root,
        data_path=data_path,
        base_results=base_jsonl,
        output_path=selector_mcq,
        model=resolved_selector_all,
        backend=qwen_backend,
        question_filter="mcq",
        mode="repair",
        batch_size=repair_batch_size,
        max_tokens=repair_max_tokens,
        max_model_len=repair_max_model_len,
        base_tail_chars=repair_base_tail_chars,
        gpu_id=gpu_id,
        resume=resume,
        reuse_existing=reuse_existing,
        boxed_prefill=boxed_prefill,
    )
    _run_qwen_repair_stage(
        root=root,
        data_path=data_path,
        base_results=base_jsonl,
        output_path=selector_free,
        model=resolved_selector_free,
        backend=qwen_backend,
        question_filter="freeform",
        mode="repair",
        batch_size=repair_batch_size,
        max_tokens=repair_max_tokens,
        max_model_len=repair_max_model_len,
        base_tail_chars=repair_base_tail_chars,
        gpu_id=gpu_id,
        resume=resume,
        reuse_existing=reuse_existing,
        boxed_prefill=boxed_prefill,
    )

    selector_jsonl = work_dir / "03_selector_row_type_merge.jsonl"
    selector_records = _merge_row_type_outputs(
        items,
        _load_jsonl_records(selector_mcq),
        _load_jsonl_records(selector_free),
        selector_jsonl,
        "selector_allrepair_mcq_freeformrepair",
    )

    if hybrid_final_policy == "selector":
        final_jsonl = work_dir / "04_final_selector_allrepair_mcq_freeformrepair.jsonl"
        final_records, safe_cleanup_changed = _apply_safe_final_string_cleanup(selector_records)
        _write_jsonl_records(final_jsonl, final_records)
        sha256 = _write_submission_csv(output_csv, items, final_records)
        summary = {
            "pipeline": TUNED_HYBRID_PRESET,
            "hybrid_final_policy": hybrid_final_policy,
            "submission_csv": str(output_csv),
            "results_jsonl": str(final_jsonl),
            "sha256": sha256,
            "rows": len(final_records),
            "expected_rows": len(items),
            "ids_match": [int(row["id"]) for row in final_records] == [int(item["id"]) for item in items],
            "blank_responses": sum(1 for row in final_records if not str(row.get("response", "")).strip()),
            "safe_cleanup_changed": safe_cleanup_changed,
            "base_model": base_model,
            "selector_all_repair_model": selector_all_repair_model,
            "selector_freeform_repair_model": selector_freeform_repair_model,
            "mcq_repair_model": mcq_repair_model,
            "freeform_structured_model": freeform_structured_model,
            "resolved_models": {
                "selector_all_repair": resolved_selector_all,
                "selector_freeform_repair": resolved_selector_free,
                "mcq_repair": "not_used_by_selector_policy",
                "freeform_structured": "not_used_by_selector_policy",
            },
            "stage_outputs": {
                "base_seed": str(base_jsonl),
                "selector_mcq": str(selector_mcq),
                "selector_freeform": str(selector_free),
                "selector_merge": str(selector_jsonl),
                "final_merge": str(final_jsonl),
            },
            "routing": {
                "selector": "MCQ rows use all-repair adapter; freeform rows use freeform-repair adapter.",
                "final": "Default conservative policy stops at the selector stage because it is more stable under fresh regeneration.",
            },
            "legal_surface": (
                "Model stages use Qwen/Qwen3-4B-Thinking-2507 or LoRA adapters "
                "whose adapter_config.json names that exact base. The remaining "
                "code handles row-type routing, JSONL/CSV packaging, and "
                "response-string trimming/LaTeX wrapper cleanup."
            ),
            "generation_parameters": {
                "base_backend": backend,
                "repair_backend": qwen_backend,
                "base_max_tokens": base_max_tokens,
                "base_max_model_len": base_max_model_len,
                "prompt_style": prompt_style,
                "temperature": temperature,
                "seed": seed,
                "disable_thinking": disable_thinking,
                "repair_max_tokens": repair_max_tokens,
                "repair_max_model_len": repair_max_model_len,
                "repair_base_tail_chars": repair_base_tail_chars,
                "boxed_prefill": boxed_prefill,
            },
            "selector_rows": len(selector_records),
        }
        summary_path = work_dir / "final_single_entry_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    final_mcq = work_dir / "04_final_mcq_e5_repair.jsonl"
    final_free = work_dir / "05_final_freeform_structured_e8.jsonl"
    _run_qwen_repair_stage(
        root=root,
        data_path=data_path,
        base_results=selector_jsonl,
        output_path=final_mcq,
        model=resolved_mcq,
        backend=qwen_backend,
        question_filter="mcq",
        mode="repair",
        batch_size=repair_batch_size,
        max_tokens=repair_max_tokens,
        max_model_len=repair_max_model_len,
        base_tail_chars=repair_base_tail_chars,
        gpu_id=gpu_id,
        resume=resume,
        reuse_existing=reuse_existing,
        boxed_prefill=boxed_prefill,
    )
    _run_qwen_repair_stage(
        root=root,
        data_path=data_path,
        base_results=selector_jsonl,
        output_path=final_free,
        model=resolved_freeform,
        backend=qwen_backend,
        question_filter="freeform",
        mode="structured",
        batch_size=repair_batch_size,
        max_tokens=repair_max_tokens,
        max_model_len=repair_max_model_len,
        base_tail_chars=repair_base_tail_chars,
        gpu_id=gpu_id,
        resume=resume,
        reuse_existing=reuse_existing,
        boxed_prefill=boxed_prefill,
    )

    final_jsonl = work_dir / "06_final_hybrid_mcq_e5_freeform_structured_e8.jsonl"
    final_records = _merge_row_type_outputs(
        items,
        _load_jsonl_records(final_mcq),
        _load_jsonl_records(final_free),
        final_jsonl,
        "hybrid_mcq_e5_freeform_structured_e8",
    )
    final_records, safe_cleanup_changed = _apply_safe_final_string_cleanup(final_records)
    _write_jsonl_records(final_jsonl, final_records)
    sha256 = _write_submission_csv(output_csv, items, final_records)

    summary = {
        "pipeline": TUNED_HYBRID_PRESET,
        "hybrid_final_policy": hybrid_final_policy,
        "submission_csv": str(output_csv),
        "results_jsonl": str(final_jsonl),
        "sha256": sha256,
        "rows": len(final_records),
        "expected_rows": len(items),
        "ids_match": [int(row["id"]) for row in final_records] == [int(item["id"]) for item in items],
        "blank_responses": sum(1 for row in final_records if not str(row.get("response", "")).strip()),
        "safe_cleanup_changed": safe_cleanup_changed,
        "base_model": base_model,
        "selector_all_repair_model": selector_all_repair_model,
        "selector_freeform_repair_model": selector_freeform_repair_model,
        "mcq_repair_model": mcq_repair_model,
        "freeform_structured_model": freeform_structured_model,
        "resolved_models": {
            "selector_all_repair": resolved_selector_all,
            "selector_freeform_repair": resolved_selector_free,
            "mcq_repair": resolved_mcq,
            "freeform_structured": resolved_freeform,
        },
        "stage_outputs": {
            "base_seed": str(base_jsonl),
            "selector_mcq": str(selector_mcq),
            "selector_freeform": str(selector_free),
            "selector_merge": str(selector_jsonl),
            "final_mcq": str(final_mcq),
            "final_freeform": str(final_free),
            "final_merge": str(final_jsonl),
        },
        "routing": {
            "selector": "MCQ rows use all-repair adapter; freeform rows use freeform-repair adapter.",
            "final": "MCQ rows use mcq-e5 adapter; freeform rows use freeform-structured-e8 adapter.",
        },
        "legal_surface": (
            "Model stages use Qwen/Qwen3-4B-Thinking-2507 or LoRA adapters "
            "whose adapter_config.json names that exact base. The remaining "
            "code handles row-type routing, JSONL/CSV packaging, and "
            "response-string trimming/LaTeX wrapper cleanup."
        ),
        "generation_parameters": {
            "base_backend": backend,
            "repair_backend": qwen_backend,
            "base_max_tokens": base_max_tokens,
            "base_max_model_len": base_max_model_len,
            "prompt_style": prompt_style,
            "temperature": temperature,
            "seed": seed,
            "disable_thinking": disable_thinking,
            "repair_max_tokens": repair_max_tokens,
            "repair_max_model_len": repair_max_model_len,
            "repair_base_tail_chars": repair_base_tail_chars,
            "boxed_prefill": boxed_prefill,
        },
        "selector_rows": len(selector_records),
    }
    summary_path = work_dir / "final_single_entry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_inference(
    data_path: str | Path = "kaggle_data/private.jsonl",
    output_csv: str | Path = "submission.csv",
    work_dir: str | Path = "results/final_pipeline",
    model: str = "Qwen/Qwen3-4B-Thinking-2507",
    repair_model: str | None = None,
    backend: str = "vllm",
    gpu_id: str | None = None,
    base_max_tokens: int = 16384,
    base_max_model_len: int = 32768,
    prompt_style: str = "cot",
    temperature: float = 0.7,
    seed: int = 42,
    disable_thinking: bool = False,
    batch_size: int = 4,
    vllm_batch_size: int = 1024,
    vllm_enforce_eager: bool = True,
    self_consistency_n: int = 1,
    model_postprocess: bool = False,
    legal_postprocess_modes: str = "repair,structured,extract,format",
    repair_max_tokens: int = 512,
    repair_max_model_len: int = 8192,
    repair_base_tail_chars: int = 2400,
    repair_disable_thinking: bool = True,
    resume: bool = True,
    reuse_existing: bool = False,
    pipeline: str = "single_model",
    selector_all_repair_model: str | None = None,
    selector_freeform_repair_model: str | None = None,
    mcq_repair_model: str | None = None,
    freeform_structured_model: str | None = None,
    repair_backend: str = "auto",
    repair_batch_size: int = 24,
    boxed_prefill: bool = False,
    hybrid_final_policy: str = "selector",
) -> dict:
    """Run the selected Qwen generation pipeline and write a Kaggle CSV.

    The default path uses Qwen/Qwen3-4B-Thinking-2507 with
    base_max_tokens=16384. The thinking flag is a generation parameter; both
    enabled and disabled thinking are supported.

    `reuse_existing=True` only skips stages whose expected intermediate files
    already exist in `work_dir`; leave it False for a clean verification run.

    The output CSV contains the final model responses in input id order.
    """
    if pipeline == RAW_DUAL_CHOOSE_PRESET:
        if self_consistency_n != 1:
            raise ValueError("raw_dual_choose requires self_consistency_n=1")
        root = Path(__file__).resolve().parent
        data_path = Path(data_path)
        if not data_path.is_absolute():
            data_path = root / data_path
        choose_work_dir = Path(work_dir)
        if not choose_work_dir.is_absolute():
            choose_work_dir = root / choose_work_dir
        choose_work_dir.mkdir(parents=True, exist_ok=True)
        t06_dir = choose_work_dir / "raw_t06"
        t08_dir = choose_work_dir / "raw_t08"
        t06_summary = run_inference(
            data_path=data_path,
            output_csv=t06_dir / "submission.csv",
            work_dir=t06_dir / "work",
            model=model,
            repair_model=repair_model,
            backend=backend,
            gpu_id=gpu_id,
            base_max_tokens=base_max_tokens,
            base_max_model_len=base_max_model_len,
            prompt_style=prompt_style,
            temperature=0.6,
            seed=seed,
            disable_thinking=disable_thinking,
            batch_size=batch_size,
            vllm_batch_size=vllm_batch_size,
            self_consistency_n=1,
            model_postprocess=False,
            resume=resume,
            reuse_existing=reuse_existing,
            pipeline="single_model",
        )
        t08_summary = run_inference(
            data_path=data_path,
            output_csv=t08_dir / "submission.csv",
            work_dir=t08_dir / "work",
            model=model,
            repair_model=repair_model,
            backend=backend,
            gpu_id=gpu_id,
            base_max_tokens=base_max_tokens,
            base_max_model_len=base_max_model_len,
            prompt_style=prompt_style,
            temperature=0.8,
            seed=seed,
            disable_thinking=disable_thinking,
            batch_size=batch_size,
            vllm_batch_size=vllm_batch_size,
            self_consistency_n=1,
            model_postprocess=False,
            resume=resume,
            reuse_existing=reuse_existing,
            pipeline="single_model",
        )
        choose_jsonl = choose_work_dir / "qwen_choose_t06_t08.jsonl"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "qwen.py"),
            "--model",
            model,
            "--backend",
            "vllm" if backend in {"auto", "vllm"} else "transformers",
            "--data",
            str(data_path),
            "--base-results",
            str(t06_summary["results_jsonl"]),
            "--candidate-results",
            f"temp08={t08_summary['results_jsonl']}",
            "--output",
            str(choose_jsonl),
            "--mode",
            "choose",
            "--max-tokens",
            str(repair_max_tokens),
            "--max-model-len",
            str(repair_max_model_len),
            "--base-tail-chars",
            str(repair_base_tail_chars),
            "--candidate-tail-chars",
            "1200",
            "--batch-size",
            str(repair_batch_size),
            "--boxed-prefill",
            "--resume",
        ]
        if gpu_id is not None:
            cmd.extend(["--gpu-id", str(gpu_id)])
        subprocess.run(cmd, check=True)

        choose_records = _load_jsonl_records(choose_jsonl)
        output_csv = Path(output_csv)
        if not output_csv.is_absolute():
            output_csv = Path(__file__).resolve().parent / output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        items = _load_jsonl_records(Path(data_path))
        by_id = {int(row["id"]): row for row in choose_records}
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow(["id", "response"])
            for item in items:
                record = by_id[int(item["id"])]
                writer.writerow([record["id"], record.get("response", "")])
        summary = {
            "results_jsonl": str(choose_jsonl),
            "submission_csv": str(output_csv),
            "sha256": hashlib.sha256(output_csv.read_bytes()).hexdigest(),
            "rows": len(choose_records),
            "expected_rows": len(items),
            "ids_match": [int(row["id"]) for row in choose_records] == [int(item["id"]) for item in items],
            "blank_responses": sum(1 for row in choose_records if not str(row.get("response", "")).strip()),
            "model": model,
            "backend": backend,
            "pipeline": [
                "qwen_generation_t0.6_raw16k",
                "qwen_generation_t0.8_raw16k",
                "qwen_model_only_choose_between_qwen_candidates",
                "csv_packaging",
            ],
            "inference_mode": "qwen_model_only",
            "public_smoke_reference": "optional public smoke configuration",
            "candidate_summaries": {
                "temp06": t06_summary,
                "temp08": t08_summary,
            },
        }
        summary_path = choose_work_dir / "raw_dual_choose_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    if pipeline == TUNED_HYBRID_PRESET:
        return _run_tuned_hybrid_pipeline(
            data_path=Path(data_path),
            output_csv=Path(output_csv),
            work_dir=Path(work_dir),
            base_model=model,
            selector_all_repair_model=(
                _required_optional_model(
                    selector_all_repair_model
                    or os.environ.get("CSE151B_SELECTOR_ALL_REPAIR_MODEL")
                    or DEFAULT_HYBRID_ADAPTERS["selector_all_repair"],
                    "CSE151B_SELECTOR_ALL_REPAIR_MODEL",
                )
            ),
            selector_freeform_repair_model=(
                _required_optional_model(
                    selector_freeform_repair_model
                    or os.environ.get("CSE151B_SELECTOR_FREEFORM_REPAIR_MODEL")
                    or DEFAULT_HYBRID_ADAPTERS["selector_freeform_repair"],
                    "CSE151B_SELECTOR_FREEFORM_REPAIR_MODEL",
                )
            ),
            mcq_repair_model=(
                mcq_repair_model
                or os.environ.get("CSE151B_MCQ_REPAIR_MODEL")
                or DEFAULT_HYBRID_ADAPTERS["mcq_repair"]
                or "Qwen/Qwen3-4B-Thinking-2507"
            ),
            freeform_structured_model=(
                freeform_structured_model
                or os.environ.get("CSE151B_FREEFORM_STRUCTURED_MODEL")
                or DEFAULT_HYBRID_ADAPTERS["freeform_structured"]
                or "Qwen/Qwen3-4B-Thinking-2507"
            ),
            backend=backend,
            gpu_id=gpu_id,
            base_max_tokens=base_max_tokens,
            base_max_model_len=base_max_model_len,
            prompt_style=prompt_style,
            temperature=temperature,
            seed=seed,
            disable_thinking=disable_thinking,
            batch_size=batch_size,
            vllm_batch_size=vllm_batch_size,
            repair_backend=repair_backend,
            repair_batch_size=repair_batch_size,
            repair_max_tokens=repair_max_tokens,
            repair_max_model_len=repair_max_model_len,
            repair_base_tail_chars=repair_base_tail_chars,
            resume=resume,
            reuse_existing=reuse_existing,
            boxed_prefill=boxed_prefill,
            hybrid_final_policy=hybrid_final_policy,
        )
    if pipeline != "single_model":
        raise ValueError(
            f"Unknown pipeline {pipeline!r}. Use {TUNED_HYBRID_PRESET!r} "
            f"or {RAW_DUAL_CHOOSE_PRESET!r} for optional experiments, "
            "or 'single_model' for the raw one-model path."
        )

    if self_consistency_n != 1:
        raise ValueError(
            "Final legal mode requires self_consistency_n=1. "
            "Do not majority-vote or select among generated answers in the final entry point."
        )
    root = Path(__file__).resolve().parent
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = root / data_path
    output_csv = Path(output_csv)
    if not output_csv.is_absolute():
        output_csv = root / output_csv
    work_dir = Path(work_dir)
    if not work_dir.is_absolute():
        work_dir = root / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    _validate_competition_model(model)
    if repair_model is None:
        repair_model = model
    _validate_competition_model(repair_model)

    model_slug = model.split("/")[-1].lower().replace("-", "_")
    repair_slug = repair_model.split("/")[-1].lower().replace("-", "_")
    base_jsonl = work_dir / f"{model_slug}_{prompt_style}_private.jsonl"
    final_jsonl = base_jsonl
    summary_path = work_dir / "final_pipeline_summary.json"
    postprocess_stage_paths: list[str] = []

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if reuse_existing and base_jsonl.exists():
        records = _load_jsonl_records(base_jsonl)
        items = _load_jsonl_records(data_path)
    else:
        items = _load_jsonl_records(data_path)
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-4B-Thinking-2507",
                trust_remote_code=True,
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        args = argparse.Namespace(
            model=model,
            backend=backend,
            max_tokens=base_max_tokens,
            max_model_len=base_max_model_len,
            dtype="auto",
            vllm_quantization="none",
            vllm_batch_size=vllm_batch_size,
            vllm_max_num_seqs=256,
            vllm_max_num_batched_tokens=32768,
            vllm_enforce_eager=vllm_enforce_eager,
            temperature=temperature,
            seed=seed,
            batch_size=batch_size,
            prompt_style=prompt_style,
            disable_thinking=disable_thinking,
            self_consistency_n=self_consistency_n,
            time_limit=0,
        )

        backend_to_use = backend
        if is_lora_adapter(model) and backend_to_use in {"auto", "vllm"}:
            backend_to_use = "transformers"
        if backend_to_use == "auto":
            try:
                responses = run_vllm(items, tokenizer, args)
                backend_to_use = "vllm"
            except Exception as exc:
                print(f"vLLM failed ({exc}), falling back to Transformers...", flush=True)
                responses = run_transformers(items, tokenizer, args)
                backend_to_use = "transformers"
        elif backend_to_use == "vllm":
            responses = run_vllm(items, tokenizer, args)
        else:
            responses = run_transformers(items, tokenizer, args)

        records = [
            {
                "id": item["id"],
                "is_mcq": bool(item.get("options")),
                "response": response,
            }
            for item, response in zip(items, responses)
        ]
        _write_jsonl_records(base_jsonl, records)

    safe_cleanup_changed = 0
    if model_postprocess:
        postprocess_modes = [
            mode.strip()
            for mode in legal_postprocess_modes.split(",")
            if mode.strip()
        ]
        allowed_postprocess_modes = {
            "repair",
            "structured",
            "extract",
            "format",
            "solve",
            "diagnose",
            "specialized",
        }
        unknown_modes = sorted(set(postprocess_modes) - allowed_postprocess_modes)
        if unknown_modes:
            raise ValueError(f"Unknown legal postprocess mode(s): {', '.join(unknown_modes)}")
        if not postprocess_modes:
            raise ValueError("legal_postprocess_modes must include at least one model prompt mode")

        repair_backend = "vllm" if backend in {"auto", "vllm"} else "transformers"
        previous_results = base_jsonl
        common_env = os.environ.copy()
        if gpu_id is not None:
            common_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        for stage_index, mode in enumerate(postprocess_modes, start=1):
            stage_output = work_dir / (
                f"model_postprocess_{stage_index:02d}_{mode}_{repair_slug}_private.jsonl"
            )
            if reuse_existing and stage_output.exists():
                print(f"Reusing {stage_output}", flush=True)
            else:
                cmd = [
                    sys.executable,
                    str(root / "qwen.py"),
                    "--model",
                    repair_model,
                    "--backend",
                    repair_backend,
                    "--data",
                    str(data_path),
                    "--base-results",
                    str(previous_results),
                    "--output",
                    str(stage_output),
                    "--question-filter",
                    "all",
                    "--batch-size",
                    "24" if repair_backend == "vllm" else "2",
                    "--max-tokens",
                    str(repair_max_tokens),
                    "--max-model-len",
                    str(repair_max_model_len),
                    "--base-tail-chars",
                    str(repair_base_tail_chars),
                    "--mode",
                    mode,
                    "--no-score",
                ]
                if repair_disable_thinking:
                    cmd.append("--disable-thinking")
                if gpu_id is not None:
                    cmd.extend(["--gpu-id", str(gpu_id)])
                if resume:
                    cmd.append("--resume")
                print("+ " + " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True, env=common_env)
            previous_results = stage_output
            postprocess_stage_paths.append(str(stage_output))

        final_jsonl = work_dir / "model_only_postprocessed_private.jsonl"
        records = _load_jsonl_records(previous_results)
        records, safe_cleanup_changed = _apply_safe_final_string_cleanup(records)
        _write_jsonl_records(final_jsonl, records)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        by_id = {int(row["id"]): row for row in records}
        for item in items:
            record = by_id[int(item["id"])]
            writer.writerow([record["id"], record.get("response", "")])

    summary = {
        "results_jsonl": str(final_jsonl),
        "submission_csv": str(output_csv),
        "sha256": hashlib.sha256(output_csv.read_bytes()).hexdigest(),
        "rows": len(records),
        "expected_rows": len(items),
        "ids_match": [int(row["id"]) for row in records] == [int(item["id"]) for item in items],
        "blank_responses": sum(1 for row in records if not str(row.get("response", "")).strip()),
        "safe_cleanup_changed": safe_cleanup_changed,
    }
    summary.update(
        {
            "model": model,
            "repair_model": repair_model,
            "backend": backend,
            "backend_used": backend_to_use if "backend_to_use" in locals() else backend,
            "pipeline": [
                f"qwen_generation_{model_slug}_{prompt_style}_t{temperature}_{base_max_tokens}",
                (
                    f"qwen_model_postprocess_{legal_postprocess_modes}_{repair_slug}"
                    f"_t0_{repair_max_tokens}_tail{repair_base_tail_chars}"
                    if model_postprocess
                    else "model_postprocess_disabled"
                ),
                "trim_and_latex_wrapper_cleanup_only",
                "csv_packaging",
            ],
            "inference_mode": "qwen_model_only",
            "base_max_tokens": base_max_tokens,
            "base_max_model_len": base_max_model_len,
            "prompt_style": prompt_style,
            "temperature": temperature,
            "seed": seed,
            "disable_thinking": disable_thinking,
            "self_consistency_n": self_consistency_n,
            "model_postprocess": model_postprocess,
            "legal_postprocess_modes": legal_postprocess_modes,
            "repair_max_tokens": repair_max_tokens,
            "repair_max_model_len": repair_max_model_len,
            "repair_base_tail_chars": repair_base_tail_chars,
            "repair_disable_thinking": repair_disable_thinking,
            "postprocess_stage_paths": postprocess_stage_paths,
            "postprocessing": (
                "Allowed-model generation passes plus response-string-only "
                "trim/LaTeX wrapper cleanup."
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default="kaggle_data/private.jsonl", help="Input JSONL path")
    parser.add_argument("--output",   default="results/public_results.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Hugging Face model name or path to a LoRA adapter directory.",
    )
    parser.add_argument("--backend",  choices=["vllm", "transformers", "auto"], default="auto",
                        help="Inference backend. 'auto' tries vLLM first, falls back to Transformers.")
    parser.add_argument("--max-tokens",     type=int, default=8192,
                        help="Max new tokens per response. Thinking model needs 4K-8K minimum.")
    parser.add_argument("--max-model-len",  type=int, default=16384)
    parser.add_argument("--dtype", default="auto",
                        help="vLLM dtype, e.g. auto, bfloat16, float16.")
    parser.add_argument("--vllm-quantization", default="bitsandbytes",
                        choices=["none", "bitsandbytes", "awq", "gptq"],
                        help="vLLM quantization mode. Use 'none' on larger GPUs.")
    parser.add_argument("--vllm-batch-size", type=int, default=128,
                        help="Number of prompts to send to vLLM per checkpointed chunk.")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=256,
                        help="vLLM max_num_seqs engine setting.")
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=32768,
                        help="vLLM max_num_batched_tokens engine setting.")
    parser.add_argument("--vllm-enforce-eager", action="store_true",
                        help="Disable vLLM torch.compile/CUDA graph capture for more reliable reproduction.")
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--batch-size",     type=int, default=4,
                        help="Batch size for Transformers backend")
    parser.add_argument("--no-score", action="store_true",
                        help="Skip scoring (use for private test set)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip questions already in the output file")
    parser.add_argument("--gpu-id",   default=None,
                        help="Set CUDA_VISIBLE_DEVICES. If omitted, keep the caller's environment.")
    parser.add_argument("--prompt-style", choices=["cot", "fewshot", "direct", "strict", "compact_boxed", "final_boxed"], default="cot",
                        help="Prompt format. 'cot' = baseline; 'fewshot' = prepend solved examples; 'direct' = concise answer.")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Ask Qwen chat templates to disable thinking mode when supported.")
    parser.add_argument("--question-filter", choices=["all", "mcq", "freeform"], default="all",
                        help="Optionally run only multiple-choice or only free-form questions.")
    parser.add_argument("--time-limit", type=float, default=0,
                        help="Time limit in minutes before automatically stopping inference (0 = no limit).")
    parser.add_argument("--self-consistency-n", type=int, default=1,
                        help="If >1, sample N responses per prompt and majority-vote on the answer (vLLM only).")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only process the first N questions (for smoke tests).")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Load data
    data = [json.loads(line) for line in open(args.data)]
    if args.question_filter != "all":
        want_mcq = args.question_filter == "mcq"
        before = len(data)
        data = [d for d in data if bool(d.get("options")) == want_mcq]
        print(f"--question-filter {args.question_filter}: kept {len(data)} / {before} questions")
    if args.limit > 0:
        data = data[: args.limit]
        print(f"--limit {args.limit}: truncated to first {len(data)} questions")
    print(f"Loaded {len(data)} questions from {args.data}")
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"  MCQ: {n_mcq}  |  Free-form: {n_free}")

    # Resume: skip already-processed IDs
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    existing_results = []
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_ids.add(r["id"])
                existing_results.append(r)
        print(f"Resuming: {len(done_ids)} already done, {len(data) - len(done_ids)} remaining.")

    pending = [d for d in data if d["id"] not in done_ids]
    if not pending:
        print("Nothing to do — all questions already processed.")
        return

    # Score setup. Initialised before Transformers inference so long runs can
    # checkpoint fully scored JSONL after every generated batch.
    if not args.no_score:
        sys.path.insert(0, str(Path(__file__).parent))
        from judger import Judger
        judger = Judger(strict_extract=False)
    else:
        judger = None

    # Load tokenizer. If args.model is an adapter path, vLLM cannot use it
    # directly and Transformers reloads the adapter tokenizer inside
    # run_transformers; this only keeps prompt construction working.
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507",
            trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    new_results = []
    all_results = existing_results

    def checkpoint_batch(batch_items, batch_responses):
        nonlocal all_results
        batch_records = [
            make_record(item, response, judger)
            for item, response in zip(batch_items, batch_responses)
        ]
        new_results.extend(batch_records)
        all_results = existing_results + new_results
        write_jsonl(out_path, all_results)
        print(f"Checkpointed {len(all_results)} records → {out_path}", flush=True)

    # Run inference
    if args.backend == "auto":
        try:
            responses = run_vllm(pending, tokenizer, args, on_batch=checkpoint_batch)
        except Exception as e:
            print(f"vLLM failed ({e}), falling back to Transformers...")
            responses = run_transformers(pending, tokenizer, args, on_batch=checkpoint_batch)
    elif args.backend == "vllm":
        responses = run_vllm(pending, tokenizer, args, on_batch=checkpoint_batch)
    else:
        responses = run_transformers(pending, tokenizer, args, on_batch=checkpoint_batch)

    if not new_results:
        for item, response in tqdm(zip(pending, responses), total=len(pending), desc="Scoring"):
            new_results.append(make_record(item, response, judger))

    all_results = existing_results + new_results

    # Save
    write_jsonl(out_path, all_results)
    print(f"\nSaved {len(all_results)} records → {out_path}")

    # Print accuracy summary if scored
    if not args.no_score:
        print_accuracy_summary(all_results)


if __name__ == "__main__":
    main()
