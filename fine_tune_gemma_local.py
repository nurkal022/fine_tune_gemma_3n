#!/usr/bin/env python3
"""
Fine-tune Gemma 3 4B on Kazakh Law QA dataset using QLoRA
Optimized for Mac M1 Pro with 16GB RAM
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

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
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone()  # For causal LM, labels = input_ids
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
    
    # Load model with reduced precision for Mac
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "mps" else None,
        low_cpu_mem_usage=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
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
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading and preprocessing dataset...")
    
    # Load dataset
    raw_data = load_jsonl_dataset(dataset_path)
    print(f"Loaded {len(raw_data)} samples")
    
    # Use subset for testing (remove [:500] for full dataset)
    train_data = raw_data[:500]  # Start with 500 samples for testing
    
    # Preprocess
    train_dataset = preprocess_dataset(train_data, tokenizer, max_length=512)
    
    print("Setting up training arguments...")
    
    # Training arguments optimized for Mac M1 Pro
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Small batch size for 16GB RAM
        gradient_accumulation_steps=4,   # Effective batch size = 4
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        fp16=True,  # Use mixed precision
        dataloader_pin_memory=False,  # Disable for Mac
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard for simplicity
        gradient_checkpointing=True,  # Save memory
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
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
        
        if device == "mps":
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