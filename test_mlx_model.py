#!/usr/bin/env python3
"""
Test script for MLX fine-tuned Kazakh Law Gemma model
"""

import os

def test_mlx_model():
    """Interactive testing for MLX model"""
    
    try:
        from mlx_lm import load, generate
        print("✅ MLX доступен")
    except ImportError:
        print("❌ MLX не установлен. Установите: pip install mlx-lm")
        return
    
    if not os.path.exists("./checkpoints"):
        print("❌ Адаптеры не найдены!")
        print("Запустите сначала: python fine_tune_gemma_mlx.py")
        return
    
    print("📦 Загрузка модели...")
    try:
        # Load the fine-tuned model with adapters
        model, tokenizer = load(
            "google/gemma-3-1b-it",
            adapter_path="./checkpoints"
        )
        print("✅ Модель загружена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    print("\n" + "="*60)
    print("🏛️  КАЗАХСКИЙ ПРАВОВОЙ АССИСТЕНТ (MLX)")
    print("="*60)
    print("Задавайте вопросы по казахстанскому праву!")
    print("Напишите 'exit' для выхода\n")
    
    # Example questions
    examples = [
        "Какие документы необходимы для регистрации ТОО в Казахстане?",
        "Какова процедура расторжения трудового договора в РК?",
        "Какие налоги обязано платить ТОО в Казахстане?",
        "Какова ответственность за нарушение налогового законодательства?",
    ]
    
    print("Примеры вопросов:")
    for i, q in enumerate(examples, 1):
        print(f"{i}. {q}")
    print()
    
    while True:
        question = input("💬 Ваш вопрос: ").strip()
        
        if question.lower() in ['exit', 'выход', 'quit']:
            break
            
        if not question:
            continue
            
        print("\n🤔 Думаю...")
        
        try:
            # Format prompt for Gemma
            prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
            # Generate response
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=300
            )
            
            # Extract just the answer part
            if "<start_of_turn>model\n" in response:
                answer = response.split("<start_of_turn>model\n")[-1]
                if "<end_of_turn>" in answer:
                    answer = answer.split("<end_of_turn>")[0]
            else:
                answer = response
            
            print(f"\n📜 Ответ:\n{answer.strip()}\n")
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Ошибка при генерации: {e}")
    
    print("До свидания! 👋")

def benchmark_mlx():
    """Quick benchmark of MLX performance"""
    print("⚡ Бенчмарк производительности MLX...")
    
    try:
        from mlx_lm import load, generate
        import time
        
        model, tokenizer = load(
            "google/gemma-3-1b-it",
            adapter_path="./checkpoints" if os.path.exists("./checkpoints") else None
        )
        
        test_prompt = "Какие документы необходимы для регистрации ТОО?"
        prompt = f"<bos><start_of_turn>user\n{test_prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Warmup
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=50)
        
        # Benchmark
        start_time = time.time()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
        end_time = time.time()
        
        # Calculate tokens per second (rough estimate)
        response_tokens = len(tokenizer.encode(response)) - len(tokenizer.encode(prompt))
        tokens_per_second = response_tokens / (end_time - start_time)
        
        print(f"📊 Производительность: ~{tokens_per_second:.1f} токенов/сек")
        print(f"⏱️  Время генерации: {end_time - start_time:.2f} сек")
        
    except Exception as e:
        print(f"❌ Ошибка бенчмарка: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_mlx()
    else:
        test_mlx_model() 