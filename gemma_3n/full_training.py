#!/usr/bin/env python3
"""
Full training script for Gemma 3 4B on complete Kazakhstani law dataset
"""

import os
import json
import subprocess
import sys
from sklearn.model_selection import train_test_split

def prepare_full_dataset():
    """Prepare complete dataset for training"""
    print("📊 Подготовка ПОЛНОГО датасета...")
    
    # Load complete dataset
    dataset_path = "final_training_dataset/kazakh_law_qa_full_20250702_013138.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Файл {dataset_path} не найден!")
        return False
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Load all data
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    
    print(f"📊 Загружено {len(raw_data)} образцов из ПОЛНОГО датасета")
    
    # Format for MLX training
    mlx_data = []
    for item in raw_data:
        formatted = {
            "text": f"<bos><start_of_turn>user\n{item['instruction']}<end_of_turn>\n<start_of_turn>model\n{item['output']}<end_of_turn><eos>"
        }
        mlx_data.append(formatted)
    
    # Split into train/validation (90/10)
    train_data, valid_data = train_test_split(mlx_data, test_size=0.1, random_state=42)
    
    # Save training data
    with open("data/train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save validation data
    with open("data/valid.jsonl", 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Подготовлено {len(train_data)} образцов для обучения, {len(valid_data)} для валидации")
    print(f"📈 Увеличение: с 7786 до {len(train_data)} (+{len(train_data)-7786} образцов)")
    return True

def run_full_training():
    """Run full training with extended parameters"""
    print("🚀 ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ...")
    print("=" * 60)
    
    # Training parameters for FULL dataset
    MODEL_NAME = "google/gemma-3-4b-it"
    DATA_PATH = "data"
    BATCH_SIZE = 1
    ITERS = 3000  # Увеличено для полного обучения
    SAVE_EVERY = 200  # Сохранять чаще
    LEARNING_RATE = 1e-4  # Чуть меньше для стабильности
    GRAD_CHECKPOINT = True
    
    print(f"🔧 Параметры обучения:")
    print(f"   • Модель: {MODEL_NAME}")
    print(f"   • Итераций: {ITERS}")
    print(f"   • Learning Rate: {LEARNING_RATE}")
    print(f"   • Batch Size: {BATCH_SIZE}")
    print(f"   • Сохранение каждые: {SAVE_EVERY} итераций")
    print(f"   • Градиентные чекпоинты: {'Да' if GRAD_CHECKPOINT else 'Нет'}")
    print()
    
    # Build MLX command
    train_cmd = f"TOKENIZERS_PARALLELISM=false python -m mlx_lm.lora " \
                f"--model {MODEL_NAME} " \
                f"--train " \
                f"--iters {ITERS} " \
                f"--data {DATA_PATH} " \
                f"--adapter-path ./checkpoints_full " \
                f"--save-every {SAVE_EVERY} " \
                f"--batch-size {BATCH_SIZE} " \
                f"--learning-rate {LEARNING_RATE} "
    
    if GRAD_CHECKPOINT:
        train_cmd += "--grad-checkpoint "
    
    print("🖥️  Команда для выполнения:")
    print(train_cmd)
    print()
    
    try:
        # Create checkpoints directory
        os.makedirs("checkpoints_full", exist_ok=True)
        
        # Run training
        result = subprocess.run(train_cmd, shell=True, check=True)
        
        print("\\n✅ ПОЛНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("📁 Адаптеры сохранены в ./checkpoints_full/")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка обучения: {e}")
        return False

def estimate_training_time():
    """Estimate training time for full dataset"""
    print("⏱️  ОЦЕНКА ВРЕМЕНИ ОБУЧЕНИЯ:")
    print("=" * 50)
    
    # Based on previous runs: ~0.8 it/sec, 185 tokens/sec
    total_iterations = 3000
    seconds_per_iteration = 1.25  # Conservative estimate
    
    total_seconds = total_iterations * seconds_per_iteration
    hours = total_seconds / 3600
    
    print(f"📊 Полный датасет: 8652 записи")
    print(f"🔄 Планируемые итерации: {total_iterations}")
    print(f"⚡ Скорость: ~0.8 итераций/сек")
    print(f"⏰ Ожидаемое время: {hours:.1f} часов ({hours*60:.0f} минут)")
    print(f"🕐 Начало: сейчас")
    print(f"🕐 Окончание: через ~{hours:.1f}ч")
    print()
    
    response = input("🤔 Продолжить полное обучение? (y/n): ").strip().lower()
    return response in ['y', 'yes', 'да', '']

def main():
    """Main function"""
    print("🏛️ ПОЛНОЕ ДООБУЧЕНИЕ GEMMA 3 4B")
    print("🌟 НА ВСЁМ ДАТАСЕТЕ КАЗАХСТАНСКОГО ПРАВА")
    print("=" * 60)
    
    # Check MLX installation
    try:
        import mlx_lm
        print("✅ MLX установлен")
    except ImportError:
        print("❌ MLX не найден. Установите: pip install mlx-lm")
        return False
    
    # Estimate time
    if not estimate_training_time():
        print("🚫 Обучение отменено пользователем")
        return False
    
    # Prepare full dataset
    if not prepare_full_dataset():
        print("❌ Ошибка подготовки датасета")
        return False
    
    # Run full training
    success = run_full_training()
    
    if success:
        print("\\n🎉 УСПЕХ! Полное обучение завершено!")
        print("🧪 Протестируйте модель:")
        print("   python test_gemma_4b.py")
        print("   # Укажите новый путь: checkpoints_full")
    
    return success

if __name__ == "__main__":
    main() 