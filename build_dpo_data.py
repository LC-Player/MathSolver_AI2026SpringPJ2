"""
Scheme 3 — Phase 1: Build DPO preference data.

Strategy: For each training sample, generate TWO reasoning chains:
  - "chosen": reasoning with the correct answer as hint (should be correct)
  - "rejected": reasoning with a WRONG answer as hint (produces flawed reasoning)

Also generates rejected responses by asking the model directly (without hint),
keeping those that produce wrong final answers as extra rejected samples.

Output: train_dpo.json with fields: question, instruction, chosen, rejected
"""
import argparse
import re
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM

from utils import load_json, save_json, extract_answer

DPO_INSTRUCTION = (
    "你是一个小学数学老师。请逐步推理以下应用题，"
    "最后以\"答案是[数字]\"的格式给出最终答案。"
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


def generate_response(messages: list, model, tokenizer, device: str = "cpu") -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def generate_chosen(question: str, answer: str, model, tokenizer, device: str) -> str:
    """Generate correct reasoning by providing the correct answer as hint."""
    messages = [
        {"role": "system", "content": DPO_INSTRUCTION},
        {"role": "user", "content": f"题目：{question}\n正确答案是{answer}，请写出解题步骤。"},
    ]
    return generate_response(messages, model, tokenizer, device)


def generate_rejected(question: str, model, tokenizer, device: str) -> str:
    """Generate possibly-incorrect reasoning without hint."""
    messages = [
        {"role": "system", "content": DPO_INSTRUCTION},
        {"role": "user", "content": question},
    ]
    return generate_response(messages, model, tokenizer, device)


def make_wrong_answer(answer: str) -> str:
    """Generate a deliberately wrong answer for rejected reasoning generation."""
    import random
    try:
        if '/' in answer:
            num, den = answer.split('/')
            wrong_num = int(num) + random.choice([1, -1, 2])
            return f"{max(1, wrong_num)}/{den}"
        val = float(answer)
        if val == int(val):
            wrong_val = int(val) + random.choice([1, -1, 10, -10])
            return str(max(0, wrong_val))
        return str(round(val + random.choice([0.5, -0.5, 1.0]), 1))
    except (ValueError, ZeroDivisionError):
        return "0"


def main():
    parser = argparse.ArgumentParser(description="Build DPO preference data")
    parser.add_argument("--model_dir", default="./Qwen/Qwen2.5-0.5B-Instruct/")
    parser.add_argument("--train_path", default="train.json")
    parser.add_argument("--output_path", default="train_dpo.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir, args.device)
    train_data = load_json(args.train_path)

    if args.max_samples:
        train_data = train_data[:args.max_samples]

    dpo_data = []
    for item in tqdm(train_data, desc="Building DPO data"):
        question = item["question"]
        answer = item["answer"]

        # Chosen: reasoning with correct answer hint
        chosen = generate_chosen(question, answer, model, tokenizer, args.device)

        # Rejected strategy 1: reasoning with wrong answer hint
        wrong_answer = make_wrong_answer(answer)
        rejected_messages = [
            {"role": "system", "content": DPO_INSTRUCTION},
            {"role": "user",
             "content": f"题目：{question}\n正确答案是{wrong_answer}，请写出解题步骤。"},
        ]
        rejected = generate_response(rejected_messages, model, tokenizer, args.device)

        # Rejected strategy 2: direct generation (keep if answer is wrong)
        direct_response = generate_rejected(question, model, tokenizer, args.device)
        direct_answer = extract_answer(direct_response)
        if direct_answer.strip() != answer.strip():
            # Add as an extra DPO pair with the same chosen
            dpo_data.append({
                "question": question,
                "instruction": DPO_INSTRUCTION,
                "chosen": chosen,
                "rejected": direct_response,
            })

        dpo_data.append({
            "question": question,
            "instruction": DPO_INSTRUCTION,
            "chosen": chosen,
            "rejected": rejected,
        })

    save_json(dpo_data, args.output_path)
    print(f"Saved {len(dpo_data)} DPO preference pairs to {args.output_path}")


if __name__ == "__main__":
    main()
