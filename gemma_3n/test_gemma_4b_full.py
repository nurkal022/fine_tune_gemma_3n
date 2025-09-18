#!/usr/bin/env python3
"""
Interactive testing for fully trained Gemma 3 4B model
"""

import os
import sys
from mlx_lm import load, generate

def get_latest_checkpoint():
    """Get the latest checkpoint"""
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
    iteration = latest_file.split('_')[0]
    return f"checkpoints_full"  # MLX loads from directory, not specific file

def interactive_chat():
    """Interactive chat with trained model"""
    print("🏛️ **GEMMA 3 4B - ПОЛНАЯ МОДЕЛЬ КАЗАХСТАНСКОГО ПРАВА**")
    print("=" * 60)
    
    try:
        checkpoint_path = get_latest_checkpoint()
        if not checkpoint_path:
            print("❌ Чекпоинты не найдены!")
            return
        
        print(f"📦 Загрузка модели с чекпоинтом: {checkpoint_path}")
        
        model, tokenizer = load(
            "google/gemma-3-4b-it",
            adapter_path=checkpoint_path
        )
        
        print("✅ Модель загружена!")
        print("\n💬 Задавайте вопросы по казахстанскому праву")
        print("   (для выхода напишите 'exit')")
        print("-" * 60)
        
        while True:
            question = input("\n📝 Ваш вопрос: ").strip()
            
            if question.lower() in ['exit', 'выход', 'quit']:
                print("👋 До свидания!")
                break
            
            if not question:
                continue
            
            print("🤔 Думаю...")
            
            prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
            try:
                response = generate(
                    model, 
                    tokenizer, 
                    prompt=prompt, 
                    max_tokens=400
                )
                
                print(f"\n🤖 Ответ: {response}\n")
                print("-" * 60)
                
            except Exception as e:
                print(f"❌ Ошибка генерации: {e}")
                
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")

def benchmark_test():
    """Quick benchmark test"""
    print("⚡ **БЕНЧМАРК ТЕСТ**")
    
    checkpoint_path = get_latest_checkpoint()
    if not checkpoint_path:
        print("❌ Чекпоинты не найдены!")
        return
    
    try:
        import time
        from mlx_lm import load, generate
        
        print("📦 Загрузка модели...")
        start_time = time.time()
        
        model, tokenizer = load(
            "google/gemma-3-4b-it",
            adapter_path=checkpoint_path
        )
        
        load_time = time.time() - start_time
        print(f"✅ Загрузка: {load_time:.2f} сек")
        
        # Benchmark generation
        prompt = "<bos><start_of_turn>user\nЧто такое ТОО в Казахстане?<end_of_turn>\n<start_of_turn>model\n"
        
        start_time = time.time()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
        gen_time = time.time() - start_time
        
        tokens = len(response.split())
        speed = tokens / gen_time if gen_time > 0 else 0
        
        print(f"🚀 Генерация: {gen_time:.2f} сек")
        print(f"📊 Скорость: {speed:.1f} токенов/сек")
        print(f"📝 Ответ: {response}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            benchmark_test()
        else:
            print("Использование: python test_gemma_4b_full.py [benchmark]")
    else:
        interactive_chat()

if __name__ == "__main__":
    main() 