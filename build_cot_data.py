"""
Scheme 2 — Phase 1: Build CoT-augmented training data.

Uses the base Qwen-0.5B model to generate step-by-step reasoning chains
for each training sample. The answer is provided as a hint so the model
can work backwards to produce correct reasoning.

Output: train_cot.json with fields: id, question, answer, reasoning, instruction
"""
import argparse
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM

from utils import load_json, save_json

COT_SYSTEM_PROMPT = (
    "你是一个小学数学老师。给你一道应用题和它的正确答案，"
    "请你写出详细的、一步一步的解题推理过程。"
    "最后以\"答案是[数字]\"结束。"
)


def build_cot_prompt(question: str, answer: str) -> str:
    return (
        f"题目：{question}\n"
        f"正确答案是：{answer}\n"
        f"请写出详细的解题步骤。"
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


def generate_reasoning(question: str, answer: str, model, tokenizer,
                       device: str = "cpu") -> str:
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": build_cot_prompt(question, answer)},
    ]
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


def augment_numbers(question: str, answer: str) -> list:
    """
    Data augmentation: replace numbers in the question and compute new answer.
    Returns a list of (new_question, new_answer) pairs.
    This is a simple rule-based augmentation for demonstration.
    """
    import re

    augmented = []
    numbers = re.findall(r'[0-9]+(?:\.[0-9]+)?', question)

    if not numbers:
        return augmented

    multipliers = [2, 3, 0.5]
    for mult in multipliers:
        new_q = question
        new_a = answer
        for num_str in set(numbers):
            try:
                old_num = float(num_str)
                if '/' in answer:
                    # Fraction answer: multiply numerator only
                    parts = answer.split('/')
                    new_a = f"{int(int(parts[0]) * mult)}/{parts[1]}"
                else:
                    new_a_val = float(answer) * mult
                    if new_a_val == int(new_a_val):
                        new_a = str(int(new_a_val))
                    else:
                        new_a = str(new_a_val)
                new_num = int(old_num * mult) if old_num == int(old_num) else old_num * mult
                new_q = new_q.replace(num_str, str(new_num), 1)
            except (ValueError, ZeroDivisionError):
                continue

        if new_q != question:
            augmented.append((new_q, new_a))

    return augmented


def main():
    parser = argparse.ArgumentParser(description="Build CoT training data")
    parser.add_argument("--model_dir", default="./Qwen/Qwen2.5-0.5B-Instruct/")
    parser.add_argument("--train_path", default="train.json")
    parser.add_argument("--output_path", default="train_cot.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples (for quick testing)")
    parser.add_argument("--augment", action="store_true",
                        help="Also generate augmented data by changing numbers")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir, args.device)
    train_data = load_json(args.train_path)

    if args.max_samples:
        train_data = train_data[:args.max_samples]

    cot_data = []
    for item in tqdm(train_data, desc="Generating CoT data"):
        question = item["question"]
        answer = item["answer"]
        reasoning = generate_reasoning(question, answer, model, tokenizer, args.device)

        cot_data.append({
            "id": item["id"],
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "instruction": COT_SYSTEM_PROMPT,
        })

        # Also generate augmented variants if requested
        if args.augment:
            for aug_q, aug_a in augment_numbers(question, answer):
                aug_reasoning = generate_reasoning(
                    aug_q, aug_a, model, tokenizer, args.device
                )
                cot_data.append({
                    "id": f"{item['id']}_aug",
                    "question": aug_q,
                    "answer": aug_a,
                    "reasoning": aug_reasoning,
                    "instruction": COT_SYSTEM_PROMPT,
                })

    save_json(cot_data, args.output_path)
    print(f"Saved {len(cot_data)} CoT-augmented samples to {args.output_path}")


if __name__ == "__main__":
    main()
