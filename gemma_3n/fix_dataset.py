#!/usr/bin/env python3
"""
Fix dataset formatting for MLX training
"""

import os
import json
from sklearn.model_selection import train_test_split

def fix_dataset():
    """Fix and recreate properly formatted dataset"""
    print("🔧 Исправляю форматирование датасета...")
    
    # Load original data
    dataset_path = "final_training_dataset/kazakh_law_qa_full_20250702_013138.jsonl"
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    
    print(f"📊 Загружено {len(raw_data)} записей")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Format for MLX training with proper escaping
    mlx_data = []
    for item in raw_data:
        # Clean text and ensure proper formatting
        instruction = item['instruction'].replace('\n', ' ').replace('\r', ' ')
        output = item['output'].replace('\n', ' ').replace('\r', ' ')
        
        # Create proper MLX format
        formatted_text = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn><eos>"
        
        mlx_data.append({"text": formatted_text})
    
    # Split data
    train_data, valid_data = train_test_split(mlx_data, test_size=0.1, random_state=42)
    
    print(f"✅ Разделение: {len(train_data)} для обучения, {len(valid_data)} для валидации")
    
    # Save with proper JSON formatting
    with open("data/train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    with open("data/valid.jsonl", 'w', encoding='utf-8') as f:
        for item in valid_data:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print("✅ Датасет исправлен!")
    return True

if __name__ == "__main__":
    fix_dataset() 