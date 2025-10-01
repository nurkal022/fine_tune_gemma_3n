#!/usr/bin/env python3
"""
Fine-tune Gemma 3 4B on Kazakh Law QA dataset using QLoRA
Optimized for Mac M1 Pro with 16GB RAM
"""

import os
import sys

# Force use only GPU 0 (RTX 4070) on multi-GPU systems
# Must be set BEFORE importing torch
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Workaround for MLX import error on non-Mac systems
if sys.platform != 'darwin':
    # Disable MLX check in transformers
    os.environ['TRANSFORMERS_NO_MLX'] = '1'
    
    # Create proper mock modules with proper __spec__
    import types
    import importlib.machinery
    
    # Create spec for mlx
    mlx_spec = importlib.machinery.ModuleSpec('mlx', None)
    mlx_module = types.ModuleType('mlx')
    mlx_module.__spec__ = mlx_spec
    mlx_module.__version__ = '0.0.0'
    mlx_module.__path__ = []
    sys.modules['mlx'] = mlx_module
    
    # Create spec for mlx.core with array class
    mlx_core_spec = importlib.machinery.ModuleSpec('mlx.core', None)
    mlx_core = types.ModuleType('mlx.core')
    mlx_core.__spec__ = mlx_core_spec
    
    # Add dummy array class that transformers checks for
    class MLXArray:
        pass
    mlx_core.array = MLXArray
    
    sys.modules['mlx.core'] = mlx_core

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# Check device availability (CUDA for Linux, MPS for Mac, CPU as fallback)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def load_jsonl_dataset(file_path):
    """Load JSONL dataset"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_prompt(instruction, output):
    """Format prompt in Gemma chat format"""
    return f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn><eos>"

def preprocess_dataset(data, tokenizer, max_length=1024):
    """Preprocess dataset for training"""
    formatted_texts = []
    
    for item in data:
        formatted_text = format_prompt(item['instruction'], item['output'])
        formatted_texts.append(formatted_text)
    
    # Tokenize without converting to tensors (DataCollator will handle padding)
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    
    # Create dataset with labels = input_ids for causal LM
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"]  # For causal LM
    })
    
    return dataset

def main():
    # Model configuration
    model_name = "google/gemma-3-1b-it"  # Gemma 3 1B - optimal for M1 Pro 16GB
    dataset_path = "final_training_dataset/kazakh_law_qa_high_quality_20250702_013138.jsonl"
    output_dir = "./gemma_kazakh_law_finetuned"
    
    print("Loading tokenizer and model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with reduced precision on CPU first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Required for Gemma3
    )
    
    print("Setting up LoRA configuration...")
    
    # LoRA configuration for memory efficiency
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    # Apply LoRA to model BEFORE moving to GPU
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Move model to device AFTER applying LoRA
    if device != "cpu":
        print(f"Moving model to {device}...")
        model = model.to(device)
    
    print("Loading and preprocessing dataset...")
    
    # Load dataset
    raw_data = load_jsonl_dataset(dataset_path)
    print(f"Loaded {len(raw_data)} samples")
    
    # Use subset for testing (remove [:500] for full dataset)
    train_data = raw_data[:500]  # Start with 500 samples for testing
    
    # Preprocess
    train_dataset = preprocess_dataset(train_data, tokenizer, max_length=512)
    
    print("Setting up training arguments...")
    
    # Training arguments optimized for available hardware
    # RTX 4070 12GB can handle batch_size=2 with Gemma 1B
    batch_size = 2 if device == "cuda" else 1
    grad_accum = 4 if device == "cuda" else 4  # Effective batch = 8
    pin_memory = True if device == "cuda" else False
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        fp16=True,  # Use mixed precision
        dataloader_pin_memory=pin_memory,
        remove_unused_columns=False,
        report_to=[],  # Disable wandb/tensorboard
        gradient_checkpointing=False,  # Disabled - conflicts with LoRA in some setups
    )
    
    # Data collator for seq2seq (handles padding properly for causal LM with labels)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    print("Starting training...")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    trainer.train()
    
    print("Saving model...")
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed! Model saved to {output_dir}")
    
    # Test the model
    print("\nTesting the model...")
    test_prompt = "Какие документы необходимы для регистрации юридического лица в Казахстане?"
    
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            f"<bos><start_of_turn>user\n{test_prompt}<end_of_turn>\n<start_of_turn>model\n",
            return_tensors="pt"
        )
        
        if device in ["cuda", "mps"]:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Question: {test_prompt}")
        print(f"Answer: {response.split('<start_of_turn>model')[-1]}")

if __name__ == "__main__":
    main() 