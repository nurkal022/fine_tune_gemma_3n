#!/usr/bin/env python3
"""
Monitor training progress and test model after completion
"""

import os
import time
import subprocess
from datetime import datetime

def check_training_status():
    """Check if training is still running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'mlx_lm' in result.stdout
    except:
        return False

def get_latest_checkpoint():
    """Get the latest checkpoint directory"""
    if not os.path.exists('checkpoints_full'):
        return None
    
    # MLX saves final adapters directly
    if os.path.exists('checkpoints_full/adapters.safetensors'):
        return 'checkpoints_full'
    
    # Look for numbered checkpoints
    files = os.listdir('checkpoints_full')
    checkpoint_files = [f for f in files if f.endswith('_adapters.safetensors')]
    if not checkpoint_files:
        return None
    
    # Get latest by number
    latest_file = max(checkpoint_files, key=lambda x: int(x.split('_')[0]))
    return f"checkpoints_full"  # MLX loads from directory, not specific file

def test_model():
    """Test the trained model"""
    print("🧪 **ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ**")
    
    try:
        from mlx_lm import load, generate
        
        checkpoint_path = get_latest_checkpoint()
        if not checkpoint_path:
            print("❌ Чекпоинты не найдены")
            return
        
        print(f"📦 Загрузка модели с чекпоинтом: {checkpoint_path}")
        
        model, tokenizer = load(
            "google/gemma-3-4b-it",
            adapter_path=checkpoint_path
        )
        
        print("✅ Модель загружена!")
        
        # Test questions
        questions = [
            "Какие документы необходимы для регистрации ТОО в Казахстане?",
            "Какова процедура расторжения трудового договора в РК?",
            "Какие налоги обязано платить ТОО в Казахстане?",
            "Кто может быть учредителем ТОО в Казахстане?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. 📝 Вопрос: {question}")
            
            prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=200
            )
            
            print(f"🤖 Ответ: {response}")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

def monitor_training():
    """Monitor training progress"""
    print("🔍 **МОНИТОРИНГ ОБУЧЕНИЯ**")
    print(f"⏰ Начало мониторинга: {datetime.now().strftime('%H:%M:%S')}")
    
    while True:
        if check_training_status():
            # Check for new checkpoints
            checkpoint = get_latest_checkpoint()
            if checkpoint:
                print(f"📊 Активный чекпоинт: {checkpoint} | {datetime.now().strftime('%H:%M:%S')}")
            else:
                print(f"🔄 Обучение в процессе... | {datetime.now().strftime('%H:%M:%S')}")
            
            time.sleep(300)  # Check every 5 minutes
        else:
            print(f"✅ Обучение завершено! | {datetime.now().strftime('%H:%M:%S')}")
            break
    
    # Wait a bit for files to finalize
    time.sleep(10)
    
    # Test the model
    test_model()

if __name__ == "__main__":
    monitor_training() 