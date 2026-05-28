"""
Scheme 1: Chain-of-Thought prompting (zero-shot + few-shot).
No training required — uses the base Qwen2.5-0.5B-Instruct model directly.

Usage:
    python cot_prompting.py --mode zero_shot   # Zero-shot CoT
    python cot_prompting.py --mode few_shot    # Few-shot CoT
    python cot_prompting.py --mode both        # Both, outputs two CSVs
"""
import argparse
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM

from utils import (
    load_json, save_csv_submit, extract_answer,
    FEW_SHOT_EXAMPLES, build_few_shot_prompt
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


def predict_few_shot(question: str, model, tokenizer, device: str = "cpu") -> str:
    few_shot_text = build_few_shot_prompt(FEW_SHOT_EXAMPLES)
    messages = [
        {"role": "system", "content": FEW_SHOT_INSTRUCTION},
        {"role": "user", "content": f"示例：\n\n{few_shot_text}\n\n现在请解答下面的题目。\n题目：{question}"},
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
                  max_samples: int = None, output_path: str = None):
    model, tokenizer = load_model(model_dir, device)
    test_data = load_json(test_path)

    if max_samples:
        test_data = test_data[:max_samples]

    results = {}
    predict_fn = predict_zero_shot if mode == "zero_shot" else predict_few_shot

    for item in tqdm(test_data, desc=f"CoT {mode}"):
        qid = int(item["id"])
        question = item["question"]
        response = predict_fn(question, model, tokenizer, device)
        answer = extract_answer(response)
        results[qid] = answer

    if output_path is None:
        output_path = f"submit_cot_{mode}.csv"
    save_csv_submit(output_path, results)
    print(f"Saved {len(results)} predictions to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CoT prompting inference")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "both"],
                        default="both", help="CoT mode")
    parser.add_argument("--model_dir", default="./Qwen/Qwen2.5-0.5B-Instruct/")
    parser.add_argument("--test_path", default="test.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples for quick testing")
    args = parser.parse_args()

    if args.mode in ("zero_shot", "both"):
        print("=== Running Zero-shot CoT ===")
        run_inference(args.model_dir, args.test_path, "zero_shot",
                      args.device, args.max_samples)

    if args.mode in ("few_shot", "both"):
        print("=== Running Few-shot CoT ===")
        run_inference(args.model_dir, args.test_path, "few_shot",
                      args.device, args.max_samples)


if __name__ == "__main__":
    main()
