"""
Unified inference for all model variants.

Supports:
  - base:    Base Qwen-0.5B with CoT prompt (Scheme 1)
  - sft:     LoRA SFT model (Scheme 2 output)
  - dpo:     LoRA DPO model (Scheme 3 output)
  - sft_cot: SFT model with CoT prompt

Supports incremental CSV output and range-based processing.

Usage:
    python infer_all.py --mode base --output submit_base.csv
    python infer_all.py --mode sft --lora_path ./output/Qwen_COT --output submit_sft.csv
    python infer_all.py --mode dpo --lora_path ./output/Qwen_DPO --output submit_dpo.csv
    python infer_all.py --mode base --start 30 --stop 100  # range
    python infer_all.py --mode base --max_samples 10  # quick test
"""
import argparse
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel

from utils import load_json, save_csv_submit, extract_answer, load_csv_submit

INFERENCE_INSTRUCTION = (
    "请逐步推理以下数学应用题，写出完整的解题步骤，"
    "最后以\"答案是[数字]\"的格式给出最终答案。"
)


def load_base_model(model_dir: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map=device, torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def load_lora_model(model_dir: str, lora_path: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map=device, torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def predict(question: str, model, tokenizer, instruction: str, device: str = "cpu",
            max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response


def run_inference(model, tokenizer, test_data: list, instruction: str,
                  device: str = "cpu", output_path: str = None,
                  save_interval: int = 10) -> dict:
    if output_path is None:
        output_path = "submit.csv"

    # Resume: load existing results, skip already-processed ids
    try:
        existing = load_csv_submit(output_path)
    except FileNotFoundError:
        existing = {}

    results = dict(existing)
    new_count = 0

    with open(output_path, 'a', encoding='utf-8') as f:
        for item in tqdm(test_data, desc="Inference"):
            qid = int(item["id"])
            if qid in results:
                continue

            question = item["question"]
            response = predict(question, model, tokenizer, instruction, device)
            answer = extract_answer(response)
            answer = answer.replace('\n', ' ').replace(',', ' ').strip()
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
    parser = argparse.ArgumentParser(description="Unified inference")
    parser.add_argument("--mode", choices=["base", "sft", "dpo", "sft_cot"],
                        default="base", help="Inference mode")
    parser.add_argument("--model_dir", default="./Qwen/Qwen2.5-0.5B-Instruct/")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--test_path", default="test.json")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--start", type=int, default=0,
                        help="Start index in test.json (inclusive, default 0)")
    parser.add_argument("--stop", type=int, default=None,
                        help="Stop index in test.json (exclusive, default until end)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Flush CSV every N new predictions")
    args = parser.parse_args()

    test_data = load_json(args.test_path)
    test_data = test_data[args.start:args.stop]
    if args.max_samples:
        test_data = test_data[:args.max_samples]

    # Determine instruction
    instruction = INFERENCE_INSTRUCTION

    # Load model
    if args.mode == "base":
        model, tokenizer = load_base_model(args.model_dir, args.device)
    elif args.mode in ("sft", "dpo", "sft_cot"):
        lora_path = args.lora_path or f"./output/Qwen_{args.mode.upper()}"
        model, tokenizer = load_lora_model(args.model_dir, lora_path, args.device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    output_path = args.output or f"submit_{args.mode}.csv"
    run_inference(model, tokenizer, test_data, instruction, args.device,
                  output_path=output_path, save_interval=args.save_interval)


if __name__ == "__main__":
    main()
