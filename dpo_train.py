"""
Scheme 3 — Phase 2: DPO training on preference pairs.

Uses HuggingFace TRL's DPOTrainer to train the model to prefer
correct reasoning chains over incorrect ones.

Usage:
    python dpo_train.py --dpo_data train_dpo.json
    python dpo_train.py --dpo_data train_dpo.json --max_samples 200
"""
import argparse
import json
import torch
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer


def format_dpo_sample(item, tokenizer):
    """Format a single DPO sample into the format expected by DPOTrainer."""
    prompt = (
        f"<|im_start|>system\n{item['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{item['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "prompt": prompt,
        "chosen": item["chosen"],
        "rejected": item["rejected"],
    }


def main():
    parser = argparse.ArgumentParser(description="DPO training")
    parser.add_argument("--model_dir", default="./Qwen/Qwen2.5-0.5B-Instruct/")
    parser.add_argument("--dpo_data", default="train_dpo.json")
    parser.add_argument("--output_dir", default="./output/Qwen_DPO")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto",
                        help="Device map: 'auto' for GPU, 'cpu' for CPU-only")
    args = parser.parse_args()

    try:
        snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="./", revision="master")
    except Exception:
        print("Model download skipped")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.device == "auto" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map=args.device, torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    # LoRA config for DPO (applied before DPOTrainer)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # Load DPO data
    with open(args.dpo_data, 'r', encoding='utf-8') as f:
        dpo_raw = json.load(f)

    if args.max_samples:
        dpo_raw = dpo_raw[:args.max_samples]

    train_dataset = [format_dpo_sample(item, tokenizer) for item in dpo_raw]
    print(f"Training on {len(train_dataset)} DPO pairs")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=args.num_epochs,
        save_steps=500,
        learning_rate=args.learning_rate,
        report_to="none",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        beta=0.1,
        bf16=(args.device == "auto"),
        gradient_checkpointing=(args.device == "auto"),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
