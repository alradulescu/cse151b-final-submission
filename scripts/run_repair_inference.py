#!/usr/bin/env python3
"""Run a bounded repair pass over an existing result JSONL.

This is a local run tool. It takes a question plus a previous model
answer and asks the model to produce a concise corrected final answer. It never
uploads to Kaggle and does not create a submission CSV.

The default rendering leaves Qwen3-Thinking mode enabled. `--disable-thinking`
uses the same Qwen3-Thinking model family but asks the chat template for a
direct boxed answer, which is useful for the legal final repair pass when all
postprocessing must stay inside the reproducible model pipeline.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_inference import extract_letter  # noqa: E402


SYSTEM_REPAIR = (
    "You are a careful math answer reviewer. You will receive a problem and a "
    "previous model answer. Check the work mentally, correct it if needed, and "
    "return only the final answer inside \\boxed{}. If the problem asks for "
    "multiple blanks, parts, or sub-answers, put all requested answers in one "
    "\\boxed{} in the original order, separated by commas. Do not include "
    "reasoning outside the final answer."
)

SYSTEM_EXTRACT = (
    "You are an answer extraction engine. Do not solve the problem from scratch. "
    "Read the previous model answer and extract the final answer it most likely "
    "intended. Return exactly one answer inside \\boxed{} and no reasoning."
)

SYSTEM_FORMAT = (
    "You are a legal LaTeX and answer-string formatter. Do not solve the problem "
    "from scratch and do not change the mathematical meaning of the previous "
    "answer. Normalize only the final answer string: fix malformed LaTeX, missing "
    "boxed braces, spacing, comma separation, answer-slot order, and obvious "
    "container formatting. Return exactly one final answer inside \\boxed{}."
)

SYSTEM_SOLVE = (
    "You are a careful math solver. Solve the problem and return only the final "
    "answer inside \\boxed{}. If the problem asks for multiple blanks, parts, "
    "or sub-answers, put all requested answers in one \\boxed{} in the original "
    "order, separated by commas. Do not include reasoning outside the final "
    "answer."
)

SYSTEM_STRUCTURED = (
    "You are a careful math answer postprocessor and reviewer. You may only use "
    "your own reasoning from the supplied problem, options, and previous model "
    "answer; do not call tools, code, calculators, lookup tables, APIs, or "
    "external references. Check for common final-answer issues: wrong MCQ "
    "letter despite correct option text, "
    "missing answer slots, final boxed answer mismatch, arithmetic or algebra "
    "slips, statistics formula slips, sequence/combinatorics slips, unit/format "
    "mistakes, and overly verbose answers that need one final boxed response. "
    "Return only the corrected final answer inside one \\boxed{}. If the problem "
    "is multiple choice, return only the option letter inside \\boxed{}. If there "
    "are multiple requested blanks, include every requested answer in order, "
    "separated by commas inside the same box."
)

LEGAL_MODEL_ONLY_REPAIR_RUBRIC = """\
Legal model-only repair rubric:
- Never call tools, code, calculators, APIs, web search, lookup tables, or external references.
- Do all checking with your own model-internal math reasoning from the supplied prompt and previous answer.
- If the previous answer already contains the correct mathematical value but in the wrong format, rewrite it legally.
- For MCQ rows, map the model's intended value or conclusion to exactly one option letter. Do not output option text.
- For multi-blank rows, count every requested [ANS] slot and return all answers in the original order.
- Prefer the final mathematical conclusion over exploratory intermediate boxed values.
- Repair common legacy failure classes:
  * option-value/letter mismatch;
  * missing or extra answer slots;
  * final boxed answer contains only the last sub-answer;
  * fraction, decimal, percent, interval, tuple, unit, and LaTeX formatting mistakes;
  * sign, rounding, precision, and order-of-answers mistakes;
  * elementary arithmetic, algebra, trig, logarithm, exponential, and geometry slips;
  * statistics formula slips for confidence intervals, test statistics, ANOVA, regression, chi-square, and sample size;
  * discrete math slips for sequences, recurrences, modular arithmetic, divisibility, graph/counting, and combinatorics;
  * answer text that should be normalized to a concise \\boxed{} response.
- When uncertain, preserve the previous answer's intended conclusion rather than inventing unsupported detail.
"""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_by_id(path: Path) -> dict[int, dict]:
    return {int(row["id"]): row for row in load_jsonl(path)}


def load_id_filter(ids: str, ids_file: Optional[Path]) -> set[int] | None:
    values: list[str] = []
    if ids:
        values.extend(part.strip() for part in ids.split(","))
    if ids_file is not None:
        for line in ids_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.extend(part.strip() for part in line.split(","))
    clean = [value for value in values if value]
    if not clean:
        return None
    return {int(value) for value in clean}


def format_options(options: Optional[list]) -> str:
    if not options:
        return ""
    labels = [chr(65 + idx) for idx in range(len(options))]
    return "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))


def build_messages(
    item: dict,
    previous_response: str,
    base_tail_chars: int,
    mode: str,
) -> list[dict]:
    options_text = format_options(item.get("options"))
    if base_tail_chars > 0 and len(previous_response) > base_tail_chars:
        previous_response = previous_response[-base_tail_chars:]
    if mode == "solve" and options_text:
        prompt = (
            "Problem:\n"
            f"{item['question']}\n\n"
            "Options:\n"
            f"{options_text}\n\n"
            "Return exactly one corrected option letter inside \\boxed{}."
        )
    elif mode == "solve":
        prompt = (
            "Problem:\n"
            f"{item['question']}\n\n"
            "Return the complete corrected final answer inside one \\boxed{}. "
            "If there are multiple requested [ANS] fields, include every field "
            "in order inside that one box."
        )
    elif mode == "extract" and options_text:
        prompt = (
            "Extract the intended multiple-choice answer from the previous answer.\n"
            "Output format must be exactly: \\boxed{A} or \\boxed{B}, etc.\n"
            "If the previous answer names an option's text but not its letter, match it to the options.\n"
            "If it ends mid-reasoning, use the strongest stated conclusion in the previous answer.\n\n"
            "Problem:\n"
            f"{item['question']}\n\n"
            "Options:\n"
            f"{options_text}\n\n"
            "Previous answer:\n"
            f"{previous_response}\n\n"
            "Only output the boxed option letter."
        )
    elif mode == "extract":
        prompt = (
            "Extract the intended final answer from the previous answer.\n"
            "Output format must be exactly one concise answer inside \\boxed{}.\n"
            "Do not recompute unless needed to identify the previous answer's final conclusion.\n\n"
            "Problem:\n"
            f"{item['question']}\n\n"
            "Previous answer:\n"
            f"{previous_response}\n\n"
            "Only output the boxed final answer."
        )
    elif mode == "format" and options_text:
        prompt = (
            "Format the previous multiple-choice answer without solving again.\n"
            "If the previous answer already contains a clear option letter, output "
            "that letter inside \\boxed{}. If it contains only the option text, "
            "match that text to the listed options. Do not change the intended "
            "choice unless the previous answer clearly says a different final "
            "letter than its own option text implies.\n\n"
            "Problem:\n"
            f"{item['question']}\n\n"
            "Options:\n"
            f"{options_text}\n\n"
            "Previous answer:\n"
            f"{previous_response}\n\n"
            "Only output the formatted boxed option letter."
        )
    elif mode == "format":
        prompt = (
            "Format the previous final answer without solving again.\n"
            "Preserve the mathematical meaning, but fix LaTeX/string issues such "
            "as malformed \\boxed{} syntax, missing braces, extra prose, answer "
            "slot separators, tuple/list/interval notation, and comma ordering "
            "when the previous answer already provides the requested values.\n\n"
            "Problem:\n"
            f"{item['question']}\n\n"
            "Previous answer:\n"
            f"{previous_response}\n\n"
            "Only output the formatted boxed final answer."
        )
    elif mode == "structured" and options_text:
        prompt = (
            f"{LEGAL_MODEL_ONLY_REPAIR_RUBRIC}\n"
            "Problem:\n"
            f"{item['question']}\n\n"
            "Options:\n"
            f"{options_text}\n\n"
            "Previous answer:\n"
            f"{previous_response}\n\n"
            "Audit the previous answer and the option mapping. If the reasoning "
            "identifies an option value but the letter is missing or wrong, output "
            "the corrected option letter. If the previous answer is wrong, solve "
            "the MCQ with your own reasoning. Output only one boxed option letter."
        )
    elif mode == "structured":
        prompt = (
            f"{LEGAL_MODEL_ONLY_REPAIR_RUBRIC}\n"
            "Problem:\n"
            f"{item['question']}\n\n"
            "Previous answer:\n"
            f"{previous_response}\n\n"
            "Audit the previous answer for missing slots, final-answer extraction "
            "errors, arithmetic/statistics/algebra/sequence/format mistakes, and "
            "repair it using only your own reasoning. Output the complete corrected "
            "final answer inside exactly one \\boxed{}."
        )
    elif options_text:
        prompt = (
            "Problem:\n"
            f"{item['question']}\n\n"
            "Options:\n"
            f"{options_text}\n\n"
            "Previous answer:\n"
            f"{previous_response}\n\n"
            "Return only the corrected option letter inside \\boxed{}."
        )
    else:
        prompt = (
            "Problem:\n"
            f"{item['question']}\n\n"
            "Previous answer:\n"
            f"{previous_response}\n\n"
            "Return the complete corrected final answer inside one \\boxed{}. "
            "If there are multiple requested [ANS] fields, include every field "
            "in order inside that one box."
        )
    return [
        {
            "role": "system",
            "content": (
                SYSTEM_SOLVE
                if mode == "solve"
                else SYSTEM_EXTRACT
                if mode == "extract"
                else SYSTEM_FORMAT
                if mode == "format"
                else SYSTEM_STRUCTURED
                if mode == "structured"
                else SYSTEM_REPAIR
            ),
        },
        {"role": "user", "content": prompt},
    ]


def render_prompt(tokenizer, messages: list[dict], disable_thinking: bool = False) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": not disable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def score_freeform(response: str, gold, judger) -> bool:
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        result = judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False
    if isinstance(result, (list, tuple)):
        return bool(result[0]) if result else False
    return bool(result)


def score_item(item: dict, response: str, judger) -> bool:
    answer = item.get("answer", item.get("gold"))
    if item.get("options"):
        return extract_letter(response, item.get("options")) == str(answer).strip().upper()
    return score_freeform(response, answer, judger)


def load_model(model_name: str, gpu_id: Optional[str]):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_path = Path(model_name)
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen3-4B-Thinking-2507")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base_model, model_name)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def load_vllm(model_name: str, args):
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    model_path = Path(model_name)
    adapter_config = model_path / "adapter_config.json"
    lora_request = None
    if adapter_config.exists():
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        load_model_name = config.get("base_model_name_or_path", "Qwen/Qwen3-4B-Thinking-2507")
        tokenizer_name = model_name
        lora_request = LoRARequest("repair_adapter", 1, str(model_path))
    else:
        load_model_name = model_name
        tokenizer_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    llm = LLM(
        model=load_model_name,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.vllm_max_num_seqs,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        enable_lora=lora_request is not None,
        max_lora_rank=args.vllm_max_lora_rank,
    )
    return llm, tokenizer, lora_request


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--data", type=Path, default=ROOT / "kaggle_data/private.jsonl")
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-filter", choices=["all", "mcq", "freeform"], default="freeform")
    parser.add_argument(
        "--ids",
        default="",
        help="Optional comma-separated question IDs to run. Applied after question-filter.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Optional file containing question IDs, comma-separated or one per line.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--vllm-max-lora-rank", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=["repair", "structured", "extract", "format", "solve"],
        default="repair",
    )
    parser.add_argument(
        "--base-tail-chars",
        type=int,
        default=2400,
        help="Keep only the tail of the previous answer to bound attention memory; 0 keeps all.",
    )
    parser.add_argument("--gpu-id")
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Render Qwen prompts with thinking disabled when supported, for direct boxed repair outputs.",
    )
    args = parser.parse_args()

    items = load_jsonl(args.data)
    base_by_id = load_by_id(args.base_results)
    if args.question_filter != "all":
        want_mcq = args.question_filter == "mcq"
        items = [item for item in items if bool(item.get("options")) == want_mcq]
    id_filter = load_id_filter(args.ids, args.ids_file)
    if id_filter is not None:
        items = [item for item in items if int(item["id"]) in id_filter]
    items = [item for item in items if int(item["id"]) in base_by_id]
    if args.start_index > 0:
        items = items[args.start_index :]
    if args.limit > 0:
        items = items[: args.limit]

    existing = []
    done_ids = set()
    if args.resume and args.output.exists():
        existing = load_jsonl(args.output)
        done_ids = {int(row["id"]) for row in existing}
    pending = [item for item in items if int(item["id"]) not in done_ids]

    print(f"Loaded {len(items)} eligible items; pending {len(pending)}")
    if not pending:
        return 0

    needs_freeform_judger = (not args.no_score) and any(not item.get("options") for item in items)
    if needs_freeform_judger:
        from judger import Judger

        judger = Judger(strict_extract=False)
    else:
        judger = None

    if args.backend == "vllm":
        from vllm import SamplingParams

        model, tokenizer, lora_request = load_vllm(args.model, args)
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
    else:
        model, tokenizer = load_model(args.model, args.gpu_id)
        sampling_params = None
        lora_request = None

    all_rows = list(existing)
    for start in tqdm(range(0, len(pending), args.batch_size), desc="Repairing"):
        batch = pending[start : start + args.batch_size]
        prompts = [
            render_prompt(
                tokenizer,
                build_messages(
                    item,
                    base_by_id[int(item["id"])]["response"],
                    args.base_tail_chars,
                    args.mode,
                ),
                disable_thinking=args.disable_thinking,
            )
            for item in batch
        ]
        if args.backend == "vllm":
            outputs = model.generate(prompts, sampling_params, lora_request=lora_request)
            responses = [out.outputs[0].text.strip() for out in outputs]
        else:
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_model_len,
            ).to(model.device)
            with __import__("torch").no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            responses = [
                tokenizer.decode(out[encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                for out in output_ids
            ]

        for item, response in zip(batch, responses):
            base_row = base_by_id[int(item["id"])]
            record = {
                "id": item["id"],
                "is_mcq": bool(item.get("options")),
                "response": response,
                "base_response": base_row.get("response", ""),
                "repair_source": str(args.base_results),
            }
            if not args.no_score:
                record["gold"] = item.get("answer", item.get("gold"))
                record["correct"] = score_item(item, response, judger)
                record["base_correct"] = score_item(item, base_row.get("response", ""), judger)
            all_rows.append(record)
        write_jsonl(args.output, sorted(all_rows, key=lambda row: int(row["id"])))

    if judger is not None:
        scored = [row for row in all_rows if "correct" in row]
        correct = sum(int(row["correct"]) for row in scored)
        base_correct = sum(int(row.get("base_correct", False)) for row in scored)
        print(
            json.dumps(
                {
                    "kaggle_upload": "forbidden_without_explicit_human_approval",
                    "rows": len(scored),
                    "repair_correct": correct,
                    "repair_score": correct / len(scored) if scored else None,
                    "base_correct_on_same_rows": base_correct,
                    "base_score_on_same_rows": base_correct / len(scored) if scored else None,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
