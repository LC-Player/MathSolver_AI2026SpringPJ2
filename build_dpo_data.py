"""
Scheme 3 — Phase 1: Build DPO preference data.

Uses DeepSeek API to generate TWO reasoning chains per sample:
  - "chosen": reasoning with the correct answer as hint (should be correct)
  - "rejected": reasoning with a WRONG answer as hint (produces flawed reasoning)

Also generates rejected responses by asking the model directly (without hint),
keeping those that produce wrong final answers as extra rejected samples.

Output: train_dpo.json with fields: question, instruction, chosen, rejected
"""
import argparse
import random
import time
import json
from openai import OpenAI
from tqdm import tqdm

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from utils import load_json, save_json, extract_answer

DPO_INSTRUCTION = (
    "你是一个小学数学老师。请逐步推理以下应用题，"
    "最后以\"答案是[数字]\"的格式给出最终答案。"
)

REJECTED_INSTRUCTION = "你是一个能够处理数学问题的智能AI助手"

REJECTED_USER_INSTRUCTION = (
    "你是一个小学数学老师，你需要逐步推理以下应用题。但你今天状态不太好，会犯错误。\n"
    "请写出看似合理但实则有错误的解题步骤。你可以有以下几种做法：\n"
    "1. 推理过程中漏掉关键步骤或简化某些计算\n"
    "2. 使用不严谨的逻辑\n"
    "3. 过程中有理解偏差，但按照这个偏差推导下去\n"
    "最后写出一个完全错误的答案，以\"答案是[数字]\"的格式结尾。\n\n"
    "生成的推理过程应该看起来像是一个参数量很低的小模型的输出。"
)


def build_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def generate_response(messages: list, client: OpenAI, max_tokens: int = 1024) -> str:
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def make_wrong_answer(answer: str) -> str:
    """Generate a deliberately wrong answer for rejected reasoning generation."""
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
    parser.add_argument("--train_path", default="train.json")
    parser.add_argument("--output_path", default="train_dpo.json")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Sleep between API calls in seconds")
    parser.add_argument("--save_interval", type=int, default=25,
                        help="Save data every N samples")
    args = parser.parse_args()

    client = build_client()
    train_data = load_json(args.train_path)

    if args.max_samples:
        train_data = train_data[:args.max_samples]

    dpo_data = []
    for sample_idx, item in enumerate(tqdm(train_data, desc="Building DPO data")):
        question = item["question"]
        answer = item["answer"]

        # Chosen: reasoning with correct answer hint
        try:
            chosen = generate_response([
                {"role": "system", "content": DPO_INSTRUCTION},
                {"role": "user",
                 "content": f"题目：{question}\n正确答案是{answer}，请写出解题步骤。"},
            ], client)
        except Exception as e:
            print(f"Error on chosen for sample {item.get('id', '?')}: {e}")
            chosen = ""

        # Rejected: reasoning with error instruction
        try:
            rejected = generate_response([
                {"role": "system", "content": REJECTED_INSTRUCTION},
                {"role": "user",
                 "content": f"{REJECTED_USER_INSTRUCTION}\n题目：{question}\n"},
            ], client, 2048)
        except Exception as e:
            print(f"Error on rejected for sample {item.get('id', '?')}: {e}")
            rejected = ""

        if chosen.strip() and rejected.strip():
            print(f"\n[{sample_idx+1}] Q: {question[:60]}...")
            print(f"    [CHOSEN] {chosen[:100]}...")
            print(f"    [REJECT] {rejected[:100]}...")
            dpo_data.append({
                "question": question,
                "instruction": DPO_INSTRUCTION,
                "chosen": chosen,
                "rejected": rejected,
            })
        else:
            print(f"\n[{sample_idx+1}] ✗ Skipped (empty chosen or rejected)")

        if len(dpo_data) % args.save_interval == 0 and len(dpo_data) > 0:
            save_json(dpo_data, args.output_path)
            print(f"    ✓ Saved {len(dpo_data)} pairs to {args.output_path}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    save_json(dpo_data, args.output_path)
    print(f"\nCompleted: {len(dpo_data)} DPO preference pairs saved to {args.output_path}")


if __name__ == "__main__":
    main()
