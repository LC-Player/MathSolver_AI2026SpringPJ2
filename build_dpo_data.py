"""
Scheme 3 — Phase 1: Build DPO preference data.

Reuses CoT-generated reasoning (from train_cot.json) as the "chosen" response,
and only calls DeepSeek API to generate "rejected" (flawed) reasoning per sample.

Supports incremental range generation: loads existing train_dpo.json,
skips already-generated ids, appends/overwrites by id.

Supports --watch mode for parallel run with build_cot_data.py.

Output: train_dpo.json — list of {id, question, instruction, chosen, rejected}
"""
import argparse
import time
import json
from openai import OpenAI
from tqdm import tqdm

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from utils import load_json, save_json, normalize_question

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
        max_tokens=2048,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def get_base_id(item_id) -> int:
    """Extract the train.json row index from a CoT data id (e.g. '123' or '123_aug' → 123)."""
    return int(str(item_id).split("_")[0])


def safe_load_json(path: str) -> list:
    """Load a JSON list, retrying on decode errors (file may be mid-write)."""
    for attempt in range(3):
        try:
            data = load_json(path)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            if attempt < 2:
                time.sleep(2)
    return load_json(path)


def load_existing_output(path: str) -> list:
    try:
        data = load_json(path)
        if isinstance(data, list):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def process_sample(item: dict, item_id: str, client: OpenAI) -> dict | None:
    """Process one CoT sample into a DPO pair. Returns None if skipped."""
    question = normalize_question(item.get("question"))
    chosen = (item.get("reasoning") or "").strip()
    instruction = (item.get("instruction") or "").strip()

    if not question:
        print(f"  Skipped: empty question (id={item_id})")
        return None

    if not chosen:
        print(f"  Skipped: empty reasoning for Q: {question[:60]}...")
        return None

    if not instruction:
        print(f"  Skipped: empty instruction for Q: {question[:60]}...")
        return None

    try:
        rejected = generate_rejected(question, client)
    except Exception as e:
        print(f"  Error on rejected (id={item_id}): {e}")
        return None

    rejected = (rejected or "").strip()
    if not rejected:
        print(f"  Skipped: empty rejected for Q: {question[:60]}...")
        return None

    print(f"  Q: {question[:60]}...")
    print(f"  [CHOSEN] {chosen[:100]}...")
    print(f"  [REJECT] {rejected[:100]}...")
    return {
        "id": item_id,
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
    parser.add_argument("--start", type=int, default=0,
                        help="Start train.json index (inclusive, default 0)")
    parser.add_argument("--stop", type=int, default=None,
                        help="Stop train.json index (exclusive, default until end)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max new samples to generate in this run")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Sleep between API calls in seconds")
    parser.add_argument("--save_interval", type=int, default=25,
                        help="Save data every N new samples")
    parser.add_argument("--watch", action="store_true",
                        help="Keep polling cot_path for new samples (parallel run with build_cot_data.py)")
    parser.add_argument("--watch_interval", type=float, default=15.0,
                        help="Seconds between polls when watch is enabled")
    parser.add_argument("--watch_retries", type=int, default=40,
                        help="Max consecutive empty polls before stopping in watch mode")
    args = parser.parse_args()

    client = build_client()

    dpo_data = load_existing_output(args.output_path)
    existing_ids = {item["id"] for item in dpo_data}
    processed_keys = set(existing_ids)  # includes failed attempts (within this session)
    new_in_run = 0
    empty_polls = 0
    range_stop = args.stop if args.stop is not None else float("inf")

    while True:
        cot_data = safe_load_json(args.cot_path)

        # Collect in-range items not yet processed
        pending = []
        for item in cot_data:
            item_id = str(item.get("id", ""))
            base_id = get_base_id(item_id)
            if not (args.start <= base_id < range_stop):
                continue
            if item_id in processed_keys:
                continue
            pending.append(item)

        # Apply max_samples limit to new items
        if args.max_samples is not None:
            remaining = args.max_samples - new_in_run
            if remaining <= 0:
                break
            pending = pending[:remaining]

        if not pending:
            if not args.watch:
                break
            empty_polls += 1
            if empty_polls >= args.watch_retries:
                print(f"\nNo new samples after {args.watch_retries} consecutive polls, stopping.")
                break
            print(f"\nNo new in-range samples (total in file={len(cot_data)}, "
                  f"in range processed={len(processed_keys)}, "
                  f"dpo pairs={len(dpo_data)}), waiting {args.watch_interval}s... "
                  f"({empty_polls}/{args.watch_retries})")
            time.sleep(args.watch_interval)
            continue

        empty_polls = 0
        print(f"\nFound {len(pending)} new sample(s) to process "
              f"(total in file={len(cot_data)}, dpo pairs={len(dpo_data)})")

        for item in tqdm(pending, desc="Building DPO data", unit="sample"):
            item_id = str(item["id"])
            processed_keys.add(item_id)

            result = process_sample(item, item_id, client)
            if result:
                dpo_data.append(result)
                existing_ids.add(item_id)
                new_in_run += 1

            if new_in_run > 0 and new_in_run % args.save_interval == 0:
                save_json(dpo_data, args.output_path)
                print(f"    Saved {len(dpo_data)} pairs to {args.output_path}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        save_json(dpo_data, args.output_path)
        print(f"Checkpoint: {len(dpo_data)} DPO pairs saved")

        if not args.watch:
            break

    save_json(dpo_data, args.output_path)
    print(f"\nCompleted: {len(dpo_data)} DPO preference pairs saved to {args.output_path}")


if __name__ == "__main__":
    main()
