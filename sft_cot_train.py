"""
Scheme 2 — Phase 2: SFT fine-tuning on CoT-augmented data.
Trains Qwen-0.5B with LoRA to produce step-by-step reasoning + answer.

Usage:
    python sft_cot_train.py                          # Full training
    python sft_cot_train.py --max_samples 500         # Quick test on 500 samples
    python sft_cot_train.py --cot_data train_cot.json # Specify data file
"""
import argparse
import json
import torch
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from utils import normalize_question


def process_func(example, tokenizer, max_length=512):
    """Preprocess a CoT-augmented sample for SFT."""
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    # Use reasoning as the target (includes the answer at the end)
    target_text = example.get("reasoning", example["answer"])
    response = tokenizer(target_text, add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="SFT training with CoT data")
    parser.add_argument("--model_dir", default="./Qwen/Qwen2.5-0.5B-Instruct/")
    parser.add_argument("--cot_data", default="train_cot.json")
    parser.add_argument("--output_dir", default="./output/Qwen_COT")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size (use 1 for CPU)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto",
                        help="Device map: 'auto' for GPU, 'cpu' for CPU-only")
    args = parser.parse_args()

    # Download model if not already cached
    try:
        snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="./", revision="master")
    except Exception:
        print("Model download skipped (may already exist or network issue)")

    torch_dtype = torch.bfloat16 if args.device == "auto" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map=args.device, torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    model.enable_input_require_grads()

    # Load CoT data
    with open(args.cot_data, 'r', encoding='utf-8') as f:
        cot_data = json.load(f)

    before = len(cot_data)
    cot_data = [d for d in cot_data
                if (d.get("reasoning") or "").strip()
                and (normalize_question(d.get("question"))).strip()
                and (d.get("instruction") or "").strip()
                and (d.get("answer") or "").strip()]
    if before > len(cot_data):
        print(f"Filtered {before - len(cot_data)} empty/incomplete samples")

    if args.max_samples:
        cot_data = cot_data[:args.max_samples]

    print(f"Training on {len(cot_data)} CoT samples")

    train_dataset = [process_func(d, tokenizer, args.max_length) for d in cot_data]

    # LoRA config
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=args.num_epochs,
        save_steps=500,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=(args.device == "auto"),
        report_to="none",
        bf16=(args.device == "auto"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
