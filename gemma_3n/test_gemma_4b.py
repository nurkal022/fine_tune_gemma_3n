#!/usr/bin/env python3
"""
Test script for fine-tuned Gemma 3 4B model with MLX
"""

import sys
from mlx_lm import load, generate

def test_model():
    """Test the fine-tuned model"""
    try:
        print("📦 Загрузка Gemma 3 4B с адаптерами...")
        
        # Load model with adapters
        model, tokenizer = load(
            "google/gemma-3-4b-it",
            adapter_path="./checkpoints"
        )
        
        print("✅ Модель загружена успешно!")
        print(f"📊 Модель: {model}")
        
        # Test questions
        questions = [
            "Какие документы необходимы для регистрации ТОО в Казахстане?",
            "Какова процедура расторжения трудового договора в РК?", 
            "Какие налоги обязано платить ТОО в Казахстане?",
            "Кто может быть учредителем ТОО по законодательству РК?"
        ]
        
        print("\n🧪 Тестирование модели на казахстанском праве...")
        print("=" * 60)
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. 📝 Вопрос: {question}")
            print("-" * 50)
            
            # Format prompt for Gemma
            prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
            try:
                # Generate response
                response = generate(
                    model, 
                    tokenizer, 
                    prompt=prompt, 
                    max_tokens=200
                )
                
                # Extract only the model's response
                if "<start_of_turn>model\n" in response:
                    answer = response.split("<start_of_turn>model\n")[-1]
                    if "<end_of_turn>" in answer:
                        answer = answer.split("<end_of_turn>")[0]
                else:
                    answer = response
                
                print(f"🤖 Ответ: {answer.strip()}")
                
            except Exception as e:
                print(f"❌ Ошибка генерации: {e}")
                continue
                
        print("\n" + "=" * 60)
        print("✅ Тестирование завершено!")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False
        
    return True

def interactive_mode():
    """Interactive chat mode"""
    try:
        print("📦 Загрузка модели для интерактивного режима...")
        model, tokenizer = load(
            "google/gemma-3-4b-it",
            adapter_path="./checkpoints"
        )
        print("✅ Готов к общению!\n")
        
        while True:
            question = input("💬 Ваш вопрос (или 'exit' для выхода): ").strip()
            
            if question.lower() in ['exit', 'quit', 'выход']:
                print("👋 До свидания!")
                break
                
            if not question:
                continue
                
            prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
            try:
                response = generate(
                    model, 
                    tokenizer, 
                    prompt=prompt, 
                    max_tokens=300
                )
                
                # Extract model response
                if "<start_of_turn>model\n" in response:
                    answer = response.split("<start_of_turn>model\n")[-1]
                    if "<end_of_turn>" in answer:
                        answer = answer.split("<end_of_turn>")[0]
                else:
                    answer = response
                    
                print(f"\n🤖 {answer.strip()}\n")
                
            except Exception as e:
                print(f"❌ Ошибка: {e}\n")
                
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        test_model() 