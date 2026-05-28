"""
Scheme 3 — Phase 1: Build DPO preference data.

Reuses CoT-generated reasoning (from train_cot.json) as the "chosen" response,
and only calls DeepSeek API to generate "rejected" (flawed) reasoning per sample.

Supports --watch mode for parallel run with build_cot_data.py:
the script polls train_cot.json and processes new samples as they appear.

Output: train_dpo.json with fields: question, instruction, chosen, rejected
"""
import argparse
import time
import json
from openai import OpenAI
from tqdm import tqdm

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from utils import load_json, save_json

REJECTED_SYSTEM_PROMPT = "你是一个能够处理数学问题的智能AI助手"

REJECTED_USER_PROMPT = (
    "你是一个小学数学老师，你需要逐步推理以下应用题。但你今天状态不太好，会犯错误。\n"
    "请写出看似合理但实则有错误的解题步骤。你可以漏掉关键步骤或简化计算、使用不严谨的逻辑、用错误的理解推导下去等\n"
    "最后写出一个完全错误的答案，以\"答案是[数字]\"的格式结尾。\n\n"
    "生成的推理过程应该看起来像是一个参数量很低的小模型的输出。"
)


def build_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def generate_rejected(question: str, client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": REJECTED_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"{REJECTED_USER_PROMPT}\n题目：{question}\n"},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def safe_load_cot(path: str):
    """Load train_cot.json, retrying on JSON decode errors (file may be mid-write)."""
    for attempt in range(3):
        try:
            return load_json(path)
        except (json.JSONDecodeError, ValueError):
            if attempt < 2:
                time.sleep(2)
    return load_json(path)


def process_sample(item: dict, sample_idx: int, client: OpenAI) -> dict | None:
    """Process one CoT sample into a DPO pair. Returns None if skipped."""
    question = (item.get("question") or "").strip()
    chosen = (item.get("reasoning") or "").strip()
    instruction = (item.get("instruction") or "").strip()

    if not question:
        print(f"\n[{sample_idx+1}] Skipped: empty question (id={item.get('id', '?')})")
        return None

    if not chosen:
        print(f"\n[{sample_idx+1}] Skipped: empty reasoning for Q: {question[:60]}...")
        return None

    if not instruction:
        print(f"\n[{sample_idx+1}] Skipped: empty instruction for Q: {question[:60]}...")
        return None

    try:
        rejected = generate_rejected(question, client)
    except Exception as e:
        print(f"\n[{sample_idx+1}] Error on rejected: {e}")
        return None

    rejected = (rejected or "").strip()
    if not rejected:
        print(f"\n[{sample_idx+1}] Skipped: empty rejected for Q: {question[:60]}...")
        return None

    print(f"\n[{sample_idx+1}] Q: {question[:60]}...")
    print(f"    [CHOSEN] {chosen[:100]}...")
    print(f"    [REJECT] {rejected[:100]}...")
    return {
        "question": question,
        "instruction": instruction,
        "chosen": chosen,
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description="Build DPO preference data from CoT data")
    parser.add_argument("--cot_path", default="train_cot.json",
                        help="Path to CoT-augmented training data")
    parser.add_argument("--output_path", default="train_dpo.json")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples total")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Sleep between API calls in seconds")
    parser.add_argument("--save_interval", type=int, default=25,
                        help="Save data every N samples")
    parser.add_argument("--watch", action="store_true",
                        help="Keep polling cot_path for new samples (parallel run with build_cot_data.py)")
    parser.add_argument("--watch_interval", type=float, default=15.0,
                        help="Seconds between polls when watch is enabled")
    parser.add_argument("--watch_retries", type=int, default=40,
                        help="Max consecutive empty polls before stopping in watch mode")
    args = parser.parse_args()

    client = build_client()

    dpo_data = []
    processed_count = 0
    empty_polls = 0

    while True:
        cot_data = safe_load_cot(args.cot_path)

        if args.max_samples and processed_count >= args.max_samples:
            break

        new_items = cot_data[processed_count:]
        if args.max_samples:
            remaining = args.max_samples - processed_count
            new_items = new_items[:remaining]

        if not new_items:
            if not args.watch:
                break
            empty_polls += 1
            if empty_polls >= args.watch_retries:
                print(f"\nNo new samples after {args.watch_retries} consecutive polls, stopping.")
                break
            print(f"\nNo new samples in {args.cot_path} (total={len(cot_data)}, "
                  f"processed={processed_count}), waiting {args.watch_interval}s... "
                  f"({empty_polls}/{args.watch_retries})")
            time.sleep(args.watch_interval)
            continue

        empty_polls = 0
        print(f"\nProcessing {len(new_items)} new sample(s) "
              f"(total in file: {len(cot_data)}, processed so far: {processed_count})")

        for item in tqdm(new_items, desc="Building DPO data", unit="sample"):
            result = process_sample(item, processed_count, client)
            processed_count += 1

            if result:
                dpo_data.append(result)

            if len(dpo_data) % args.save_interval == 0 and len(dpo_data) > 0:
                save_json(dpo_data, args.output_path)
                print(f"    Saved {len(dpo_data)} pairs to {args.output_path}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        save_json(dpo_data, args.output_path)
        print(f"Checkpoint: {len(dpo_data)} DPO pairs saved (processed {processed_count}/{len(cot_data)} CoT samples)")

        if not args.watch:
            break

    print(f"\nCompleted: {len(dpo_data)} DPO preference pairs saved to {args.output_path}")


if __name__ == "__main__":
    main()
