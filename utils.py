"""
Shared utilities: answer extraction, data I/O, model helpers.
Used by all scheme scripts to avoid duplication.
"""
import re
import json
from typing import List, Optional, Union


def normalize_question(question) -> str:
    """Normalize a question field that may be a string or a list of strings."""
    if isinstance(question, list):
        return "".join(str(p) for p in question)
    return str(question) if question else ""


def extract_answer(text: str) -> str:
    """
    Extract a numerical answer from model output.

    Strategy (tried in order):
    1. Match explicit answer markers like "答案是X" "答案为X" "结果是X"
    2. Match the last standalone number (integer, decimal, fraction) in the text
    3. Fall back to returning the raw text stripped

    Returns the answer as a string (may be integer, decimal like "7.5", or fraction like "4/5").
    """
    if not text:
        return ""

    text = text.strip()

    # Patterns that explicitly signal the final answer
    answer_patterns = [
        # Chinese markers
        r'答案\s*(?:是|为|：|:)\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)',
        r'结果\s*(?:是|为|：|:)\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)',
        r'最终答案\s*(?:是|为|：|:)\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)',
        # English markers
        r'(?:answer|result)\s*(?:is|=|:)\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)',
        # Arrow / box notation
        r'→\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)',
        r'[=＝]\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)\s*(?:[。，,.\s]|$)',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # Fallback: find all number-like tokens and return the last one
    number_pattern = r'([0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)'
    all_numbers = re.findall(number_pattern, text)

    if all_numbers:
        # Filter out unlikely answers (pure integers that look like step numbers)
        # e.g. "步骤1" "第2步"
        candidate = all_numbers[-1]
        return candidate

    return text


def load_json(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_csv_submit(path: str) -> dict:
    """Load a submit.csv, return {id: answer} mapping."""
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    return mapping


def save_csv_submit(path: str, results: dict):
    """Save results dict {id: answer} to submit CSV format."""
    with open(path, 'w', encoding='utf-8') as f:
        for idx in sorted(results.keys()):
            f.write(f"{idx},{results[idx]}\n")


def evaluate(predictions: dict, ground_truth_path: str = "train.json") -> dict:
    """
    Evaluate prediction accuracy against ground truth.
    Only works for train.json (since test.json has no labels).
    Returns {accuracy, correct, total} dict.
    """
    data = load_json(ground_truth_path)
    correct = 0
    total = 0
    for item in data:
        item_id = int(item["id"])
        if item_id in predictions:
            total += 1
            pred = predictions[item_id].strip()
            gold = item["answer"].strip()
            if pred == gold:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


# Few-shot CoT exemplars for elementary math problems
FEW_SHOT_EXAMPLES = [
    {
        "question": "商店有4框苹果，每框55千克，已经卖出135千克，还剩多少千克苹果？",
        "reasoning": """步骤：
1. 先计算总共有多少千克苹果：4框 × 55千克/框 = 220千克
2. 减去已经卖出的135千克：220 - 135 = 85千克
3. 所以还剩85千克苹果。
答案是85。"""
    },
    {
        "question": "玩具厂生产了960个电子玩具，每3个装一盒，每5盒装一箱，一共装了多少箱？",
        "reasoning": """步骤：
1. 先计算可以装多少盒：960 ÷ 3 = 320盒
2. 再计算可以装多少箱：320 ÷ 5 = 64箱
3. 所以一共装了64箱。
答案是64。"""
    },
    {
        "question": "食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？",
        "reasoning": """步骤：
1. 青菜是萝卜的3倍
2. 萝卜有105千克
3. 青菜 = 105 × 3 = 315千克
答案是315。"""
    },
    {
        "question": "田径场上，爸爸跑一圈用4分钟，妈妈跑一圈用6分钟，小红跑一圈用8分钟。爸爸和妈妈同时从起点出发，他们几分钟后可以在起点第一次相遇？",
        "reasoning": """步骤：
1. 爸爸跑一圈4分钟，妈妈跑一圈6分钟
2. 他们在起点相遇的时间是两人各跑整数圈的时间
3. 即求4和6的最小公倍数
4. 4的倍数：4, 8, 12, 16, ...
5. 6的倍数：6, 12, 18, ...
6. 最小公倍数是12
7. 所以12分钟后在起点第一次相遇。
答案是12。"""
    },
]


def build_few_shot_prompt(examples: List[dict]) -> str:
    """Build a few-shot prompt string from examples."""
    parts = []
    for ex in examples:
        parts.append(f"题目：{ex['question']}\n{ex['reasoning']}")
    return "\n\n".join(parts)
