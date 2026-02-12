"""PEFT LoRA Fine-tuning script using Transformers + PEFT.

Fine-tunes FunctionGemma (270M) with LoRA adapters on Apple Silicon.

Usage:
    python -m tools.finetune_peft --run-id 1
    python -m tools.finetune_peft --run-id 1 --epochs 3 --lr 5e-5

Requires: .venv-ft (Python 3.12 with transformers, peft, accelerate)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
PEFT_ADAPTERS_DIR = PROJECT_ROOT / "data" / "peft_adapters"

# Default hyperparameters
DEFAULT_MODEL = "google/functiongemma-270m-it"
DEFAULT_EPOCHS = 5  # More epochs for better learning
DEFAULT_BATCH_SIZE = 1  # Reduced for M1 8GB memory
DEFAULT_LR = 3e-5  # Slightly higher for faster convergence
DEFAULT_LORA_RANK = 16  # Increased for more capacity
DEFAULT_LORA_ALPHA = 32  # 2x rank for stable training
DEFAULT_MAX_LENGTH = 512  # Balanced for M1 8GB memory
DEFAULT_MAX_STEPS = 500  # More steps for thorough training


def load_training_data(run_dir: Path) -> list[dict]:
    """Load training data from JSONL file."""
    train_file = run_dir / "train.jsonl"
    data = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_conversation_for_training(example: dict, tokenizer) -> dict:
    """Convert conversation to FunctionGemma format for training.

    FunctionGemma uses a specific format:
    - System prompt with available tools
    - User/model turns with <start_of_turn>/<end_of_turn> tags
    - Function calls as JSON in model response
    """
    messages = example.get("messages", [])
    tools = example.get("tools", [])

    # Build the prompt
    parts = []

    # Add tools as system context (simplified for shorter prompts)
    if tools:
        tools_text = "Available tools:\n"
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            # Only include name and description, skip parameters to reduce length
            tools_text += f"- {name}: {desc}\n"
        parts.append(tools_text)

    # Build conversation
    response_text = ""
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            if tool_calls:
                # Last message with tool call - this is what we want to predict
                call = tool_calls[0]
                func = call.get("function", {})
                func_name = func.get("name", "")
                func_args = func.get("arguments", "{}")
                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except:
                        pass
                response_text = json.dumps({
                    "name": func_name,
                    "arguments": func_args
                }, ensure_ascii=False)
            elif content:
                if i == len(messages) - 1:
                    # Last message without tool call
                    response_text = content
                else:
                    parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

    # Add model turn start
    parts.append("<start_of_turn>model\n")

    input_text = "\n".join(parts)
    full_text = input_text + response_text + "<end_of_turn>"

    return {
        "input_text": input_text,
        "output_text": response_text,
        "full_text": full_text,
    }


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the examples for causal LM training."""
    # Tokenize full text
    model_inputs = tokenizer(
        examples["full_text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    # For causal LM, labels = input_ids (shifted internally by the model)
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    # Mask the input portion (only train on output)
    input_tokens = tokenizer(
        examples["input_text"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    # Set labels to -100 for input tokens (ignore in loss)
    input_len = len(input_tokens["input_ids"])
    model_inputs["labels"][:input_len] = [-100] * input_len

    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="PEFT LoRA Fine-tune FunctionGemma")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID (1 or 2)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max sequence length")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max training steps (0 for epoch-based)")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    run_dir = FINETUNE_DIR / f"run_{args.run_id}"
    output_dir = PEFT_ADAPTERS_DIR / f"run_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check training data exists
    if not (run_dir / "train.jsonl").exists():
        logger.error(f"Training data not found: {run_dir / 'train.jsonl'}")
        logger.error("Run split_data.py first!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"PEFT LoRA Fine-tuning (Run {args.run_id})")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Training data:  {run_dir / 'train.jsonl'}")
    print(f"  Output:         {output_dir}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Max steps:      {args.max_steps if args.max_steps > 0 else 'auto'}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  LoRA alpha:     {args.lora_alpha}")
    print(f"  Max length:     {args.max_length}")
    print("=" * 60 + "\n")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device (MPS for M1, CUDA for GPU, CPU otherwise)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load model with appropriate dtype for M1
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use float32 for M1 compatibility
    )

    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,  # Increased for regularization
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Include MLP layers
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and process training data
    logger.info("Loading training data...")
    raw_data = load_training_data(run_dir)
    logger.info(f"Loaded {len(raw_data)} training samples")

    # Format data for training
    formatted_data = [format_conversation_for_training(ex, tokenizer) for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)

    # Tokenize dataset
    logger.info("Tokenizing dataset...")

    def tokenize_batch(examples):
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["full_text"])):
            # Tokenize full text
            full_tokens = tokenizer(
                examples["full_text"][i],
                max_length=args.max_length,
                truncation=True,
                padding=False,
            )

            # Tokenize input only to find where to mask
            input_tokens = tokenizer(
                examples["input_text"][i],
                max_length=args.max_length,
                truncation=True,
                padding=False,
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]

            # Create labels with -100 for input portion
            labels = input_ids.copy()
            input_len = min(len(input_tokens["input_ids"]), len(labels))
            labels[:input_len] = [-100] * input_len

            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            results["labels"].append(labels)

        return results

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments
    max_steps = args.max_steps if args.max_steps > 0 else -1
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=max_steps,  # -1 means use epochs
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=25,  # Log every 25 steps
        save_steps=100,
        save_total_limit=3,
        fp16=False,  # Disable for M1
        bf16=False,  # Disable for M1
        gradient_accumulation_steps=4,
        warmup_steps=50,  # More warmup for stability
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,  # For M1
        gradient_checkpointing=False,  # Disabled - was causing memory issues
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save the adapter
    logger.info(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print(f"PEFT Fine-tuning Complete (Run {args.run_id})")
    print("=" * 60)
    print(f"  Adapter saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
