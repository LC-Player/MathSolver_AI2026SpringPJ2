"""
Scheme 1: Chain-of-Thought prompting (zero-shot + few-shot).
No training required — uses the base Qwen2.5-0.5B-Instruct model directly.

Supports incremental CSV output and dynamic few-shot examples from train_cot.json.

Usage:
    python cot_prompting.py --mode zero_shot
    python cot_prompting.py --mode few_shot
    python cot_prompting.py --mode few_shot --cot_path train_cot.json --num_examples 6
    python cot_prompting.py --mode few_shot --cot_path train_cot.json --dpo_path train_dpo.json --num_wrong 2
"""
import argparse
import random
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM

from utils import (
    load_json, save_csv_submit, extract_answer, load_csv_submit,
    FEW_SHOT_EXAMPLES, build_few_shot_prompt,
)

ZERO_SHOT_INSTRUCTION = (
    "请一步一步地推理这个数学应用题，写出完整的解题步骤，"
    "最后以\"答案是[数字]\"的格式给出最终答案。"
)

FEW_SHOT_INSTRUCTION = (
    "请参考下面的示例，一步一步地推理这个数学应用题，"
    "写出完整的解题步骤，最后以\"答案是[数字]\"的格式给出最终答案。"
)


def load_model(model_dir: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map=device, torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def sample_cot_examples(cot_path: str, num: int) -> list:
    """Randomly sample N valid CoT examples from train_cot.json."""
    cot_data = load_json(cot_path)
    valid = [d for d in cot_data
             if (d.get("reasoning") or "").strip() and (d.get("question") or "").strip()]
    if len(valid) <= num:
        return valid
    return random.sample(valid, num)


def sample_wrong_cases(dpo_path: str, num: int) -> list:
    """Randomly sample N wrong cases from train_dpo.json."""
    dpo_data = load_json(dpo_path)
    valid = [d for d in dpo_data
             if (d.get("rejected") or "").strip() and (d.get("question") or "").strip()]
    if len(valid) <= num:
        return valid
    return random.sample(valid, num)


def build_examples_prompt(examples: list) -> str:
    """Build a few-shot prompt from CoT examples (question + reasoning format)."""
    parts = []
    for ex in examples:
        parts.append(f"题目：{ex['question']}\n{ex['reasoning']}")
    return "\n\n".join(parts)


def build_wrong_cases_prompt(wrong_cases: list) -> str:
    """Build a warning section from wrong reasoning examples."""
    parts = ["以下是常见的错误推理示例，请仔细阅读并避免犯类似的错误：\n"]
    for i, ex in enumerate(wrong_cases, 1):
        parts.append(
            f"错误示例{i}：\n"
            f"题目：{ex['question']}\n"
            f"错误推理：{ex['rejected']}\n"
            f"（以上推理存在错误，请不要模仿，应当独立思考正确解法。）"
        )
    return "\n".join(parts)


def predict_zero_shot(question: str, model, tokenizer, device: str = "cpu") -> str:
    messages = [
        {"role": "system", "content": ZERO_SHOT_INSTRUCTION},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response


def predict_few_shot(question: str, model, tokenizer, device: str = "cpu",
                     examples: list = None, wrong_cases: list = None) -> str:
    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    few_shot_text = build_examples_prompt(examples)

    user_parts = [f"示例：\n\n{few_shot_text}"]

    if wrong_cases:
        user_parts.append("\n\n" + build_wrong_cases_prompt(wrong_cases))

    user_parts.append(f"\n\n现在请解答下面的题目。\n题目：{question}")

    messages = [
        {"role": "system", "content": FEW_SHOT_INSTRUCTION},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response


def run_inference(model_dir: str, test_path: str, mode: str, device: str = "cpu",
                  max_samples: int = None, output_path: str = None,
                  save_interval: int = 10, examples: list = None,
                  wrong_cases: list = None, start: int = 0, stop: int = None):
    model, tokenizer = load_model(model_dir, device)
    test_data = load_json(test_path)

    test_data = test_data[start:stop]
    if max_samples:
        test_data = test_data[:max_samples]

    if output_path is None:
        output_path = f"submit_cot_{mode}.csv"

    # Resume: load existing results, skip already-processed ids
    try:
        existing = load_csv_submit(output_path)
    except FileNotFoundError:
        existing = {}

    results = dict(existing)
    new_count = 0

    if mode == "zero_shot":
        predict_fn = predict_zero_shot
    else:
        predict_fn = lambda q: predict_few_shot(q, model, tokenizer, device,
                                                 examples, wrong_cases)

    with open(output_path, 'a', encoding='utf-8') as f:
        for item in tqdm(test_data, desc=f"CoT {mode}"):
            qid = int(item["id"])
            if qid in results:
                continue

            question = item["question"]
            response = predict_fn(question)
            answer = extract_answer(response)
            results[qid] = answer
            new_count += 1

            f.write(f"{qid},{answer}\n")
            if new_count % save_interval == 0:
                f.flush()
                print(f"  Saved {new_count} new predictions (total: {len(results)})")

    saved_new = len(results) - len(existing)
    print(f"Saved {saved_new} new predictions (total: {len(results)}) to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CoT prompting inference")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "both"],
                        default="few_shot", help="CoT mode")
    parser.add_argument("--model_dir", default="./Qwen/Qwen2.5-0.5B-Instruct/")
    parser.add_argument("--test_path", default="test.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples for quick testing")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index in test.json (inclusive, default 0)")
    parser.add_argument("--stop", type=int, default=None,
                        help="Stop index in test.json (exclusive, default until end)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Flush CSV every N new predictions")
    parser.add_argument("--cot_path", default=None,
                        help="Path to train_cot.json for dynamic few-shot examples")
    parser.add_argument("--num_examples", type=int, default=4,
                        help="Number of CoT examples to sample (default 4)")
    parser.add_argument("--dpo_path", default=None,
                        help="Path to train_dpo.json for wrong-case examples (optional)")
    parser.add_argument("--num_wrong", type=int, default=0,
                        help="Number of wrong cases to sample (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for example sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    # Prepare dynamic examples
    examples = None
    wrong_cases = None

    if args.cot_path:
        examples = sample_cot_examples(args.cot_path, args.num_examples)
        print(f"Sampled {len(examples)} CoT examples from {args.cot_path}")

    if args.dpo_path and args.num_wrong > 0:
        wrong_cases = sample_wrong_cases(args.dpo_path, args.num_wrong)
        print(f"Sampled {len(wrong_cases)} wrong cases from {args.dpo_path}")

    if args.mode in ("zero_shot", "both"):
        print("=== Running Zero-shot CoT ===")
        run_inference(args.model_dir, args.test_path, "zero_shot",
                      args.device, args.max_samples,
                      save_interval=args.save_interval,
                      start=args.start, stop=args.stop)

    if args.mode in ("few_shot", "both"):
        print("=== Running Few-shot CoT ===")
        mode_label = "few_shot"
        if examples:
            mode_label += f"_cot{len(examples)}"
        if wrong_cases:
            mode_label += f"_wrong{len(wrong_cases)}"
        output_path = f"submit_cot_{mode_label}.csv"
        run_inference(args.model_dir, args.test_path, "few_shot",
                      args.device, args.max_samples,
                      output_path=output_path,
                      save_interval=args.save_interval,
                      examples=examples, wrong_cases=wrong_cases,
                      start=args.start, stop=args.stop)


if __name__ == "__main__":
    main()
