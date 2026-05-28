"""
Unified inference for all model variants.

Supports:
  - base:    Base Qwen-0.5B with CoT prompt (Scheme 1)
  - sft:     LoRA SFT model (Scheme 2 output)
  - dpo:     LoRA DPO model (Scheme 3 output)
  - sft_cot: SFT model with CoT prompt

Usage:
    python infer_all.py --mode base --output submit_base.csv
    python infer_all.py --mode sft --lora_path ./output/Qwen_COT --output submit_sft.csv
    python infer_all.py --mode dpo --lora_path ./output/Qwen_DPO --output submit_dpo.csv
    python infer_all.py --mode base --max_samples 10  # quick test
"""
import argparse
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel

from utils import load_json, save_csv_submit, extract_answer

INFERENCE_INSTRUCTION = (
    "请逐步推理以下数学应用题，写出完整的解题步骤，"
    "最后以\"答案是[数字]\"的格式给出最终答案。"
)


def load_base_model(model_dir: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response


def run_inference(model, tokenizer, test_data: list, instruction: str,
                  device: str = "cpu") -> dict:
    results = {}
    for item in tqdm(test_data, desc="Inference"):
        qid = int(item["id"])
        question = item["question"]
        response = predict(question, model, tokenizer, instruction, device)
        answer = extract_answer(response)
        # Clean up: replace newlines, commas that could break CSV
        answer = answer.replace('\n', ' ').replace(',', ' ').strip()
        results[qid] = answer
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
    args = parser.parse_args()

    test_data = load_json(args.test_path)
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

    results = run_inference(model, tokenizer, test_data, instruction, args.device)

    output_path = args.output or f"submit_{args.mode}.csv"
    save_csv_submit(output_path, results)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
