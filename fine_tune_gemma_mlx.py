#!/usr/bin/env python3
"""
Fine-tune Gemma 3 1B using MLX for Mac M1 Pro
Based on: https://gist.github.com/alexweberk/635431b5c5773efd6d1755801020429f
"""

import json
import os
from pathlib import Path

def install_mlx():
    """Install MLX if not available"""
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.tuners.lora import LoRALinear
        print("✅ MLX уже установлен")
        return True
    except ImportError:
        print("📦 Устанавливаю MLX...")
        os.system("pip install mlx-lm")
        try:
            import mlx.core as mx
            from mlx_lm import load, generate
            print("✅ MLX установлен успешно")
            return True
        except ImportError:
            print("❌ Не удалось установить MLX")
            return False

def prepare_dataset():
    """Prepare dataset in MLX format"""
    print("📊 Подготовка датасета...")
    
    # Load your dataset
    dataset_path = "final_training_dataset/kazakh_law_qa_high_quality_20250702_013138.jsonl"
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Convert to MLX format
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    
    # Format for MLX training (first 500 for testing)
    mlx_data = []
    for item in raw_data[:500]:  # Start with 500 samples
        formatted = {
            "text": f"<bos><start_of_turn>user\n{item['instruction']}<end_of_turn>\n<start_of_turn>model\n{item['output']}<end_of_turn><eos>"
        }
        mlx_data.append(formatted)
    
    # Save train and valid sets
    train_size = int(len(mlx_data) * 0.9)
    train_data = mlx_data[:train_size]
    valid_data = mlx_data[train_size:]
    
    with open("data/train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open("data/valid.jsonl", 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Подготовлено {len(train_data)} образцов для обучения, {len(valid_data)} для валидации")

def run_mlx_training():
    """Run MLX LoRA training"""
    print("🚀 Запуск обучения с MLX...")
    
    # Training command for MLX
    cmd = """
TOKENIZERS_PARALLELISM=false python -m mlx_lm.lora \
    --model "google/gemma-3-1b-it" \
    --train \
    --iters 200 \
    --data data \
    --adapter-path ./checkpoints \
    --save-every 50 \
    --batch-size 1 \
    --grad-checkpoint \
    --learning-rate 2e-4
"""
    
    print("Выполняю команду:")
    print(cmd)
    
    os.system(cmd.strip())

def test_model():
    """Test the fine-tuned model"""
    print("🧪 Тестирование модели...")
    
    try:
        from mlx_lm import load, generate
        
        # Load the fine-tuned model
        model, tokenizer = load(
            "google/gemma-3-1b-it",
            adapter_path="./checkpoints"
        )
        
        # Test question
        test_prompt = "Какие документы необходимы для регистрации ТОО в Казахстане?"
        prompt = f"<bos><start_of_turn>user\n{test_prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        response = generate(model, tokenizer, prompt=prompt, max_tokens=256)
        
        print(f"\n📝 Вопрос: {test_prompt}")
        print(f"🤖 Ответ: {response}")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

def main():
    """Main function"""
    print("🏛️ ДООБУЧЕНИЕ GEMMA 3 1B С MLX")
    print("=" * 50)
    
    # Check MLX installation
    if not install_mlx():
        print("❌ Не удалось установить MLX. Используйте обычный скрипт.")
        return
    
    # Prepare dataset
    prepare_dataset()
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Run training
    run_mlx_training()
    
    # Test model
    test_model()
    
    print("\n✅ Обучение завершено!")
    print("📁 Адаптеры сохранены в ./checkpoints/")
    print("🧪 Используйте test_mlx_model.py для интерактивного тестирования")

if __name__ == "__main__":
    main() 