"""
Scheme 2 — Phase 1: Build CoT-augmented training data.

Uses DeepSeek API to generate step-by-step reasoning chains
for each training sample. The answer is provided as a hint so the model
can work backwards to produce correct reasoning.

Output: train_cot.json with fields: id, question, answer, reasoning, instruction
"""
import argparse
import time
import re
import json
from openai import OpenAI
from tqdm import tqdm

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from utils import load_json, save_json

COT_SYSTEM_PROMPT = (
    "你是一个小学数学老师。给你一道应用题和它的正确答案，"
    "请你写出详细的、一步一步的解题推理过程。"
    "最后以\"答案是[数字]\"结束。"
)


def build_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def generate_reasoning(question: str, answer: str, client: OpenAI) -> str:
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user",
         "content": f"题目：{question}\n正确答案是：{answer}\n请写出详细的解题步骤。"},
    ]
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def augment_numbers(question: str, answer: str) -> list:
    """
    Data augmentation: replace numbers in the question and compute new answer.
    Returns a list of (new_question, new_answer) pairs.
    """
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
    parser.add_argument("--train_path", default="train.json")
    parser.add_argument("--output_path", default="train_cot.json")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples (for quick testing)")
    parser.add_argument("--augment", action="store_true",
                        help="Also generate augmented data by changing numbers")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Sleep between API calls in seconds")
    parser.add_argument("--save_interval", type=int, default=25,
                        help="Save data every N samples")
    args = parser.parse_args()

    client = build_client()
    train_data = load_json(args.train_path)

    if args.max_samples:
        train_data = train_data[:args.max_samples]

    cot_data = []
    for idx, item in enumerate(tqdm(train_data, desc="Generating CoT data")):
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()

        if not question or not answer:
            print(f"\n[{idx+1}] Skipped: empty question or answer (id={item.get('id', '?')})")
            continue

        try:
            reasoning = generate_reasoning(question, answer, client)
        except Exception as e:
            print(f"\n[{idx+1}] Error on sample {item.get('id', '?')}: {e}")
            reasoning = ""

        reasoning = (reasoning or "").strip()
        if not reasoning:
            print(f"\n[{idx+1}] Skipped: empty reasoning for Q: {question[:50]}...")
            continue

        cot_data.append({
            "id": item["id"],
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "instruction": COT_SYSTEM_PROMPT,
        })

        print(f"\n[{idx+1}] Q: {question[:50]}...")
        print(f"    A: {answer}")

        if args.augment:
            for aug_q, aug_a in augment_numbers(question, answer):
                try:
                    aug_reasoning = generate_reasoning(aug_q, aug_a, client)
                except Exception as e:
                    print(f"    [AUG] Error on id={item.get('id', '?')}: {e}")
                    aug_reasoning = ""

                aug_reasoning = (aug_reasoning or "").strip()
                if not aug_reasoning:
                    print(f"    [AUG] Skipped: empty reasoning")
                    continue

                cot_data.append({
                    "id": f"{item['id']}_aug",
                    "question": aug_q,
                    "answer": aug_a,
                    "reasoning": aug_reasoning,
                    "instruction": COT_SYSTEM_PROMPT,
                })
                print(f"    [AUG] Q: {aug_q[:50]}...")
                print(f"           A: {aug_a}")

        if len(cot_data) % args.save_interval == 0 and len(cot_data) > 0:
            save_json(cot_data, args.output_path)
            print(f"    Saved {len(cot_data)} samples to {args.output_path}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    save_json(cot_data, args.output_path)
    print(f"\nCompleted: {len(cot_data)} CoT-augmented samples saved to {args.output_path}")


if __name__ == "__main__":
    main()
