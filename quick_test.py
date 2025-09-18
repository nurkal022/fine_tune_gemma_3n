#!/usr/bin/env python3
"""
Quick test of the fine-tuned MLX model
"""

def quick_test():
    try:
        from mlx_lm import load, generate
        print("📦 Загрузка модели...")
        
        model, tokenizer = load(
            "google/gemma-3-1b-it",
            adapter_path="./checkpoints"
        )
        print("✅ Модель загружена!")
        
        # Test questions
        questions = [
            "Какие документы необходимы для регистрации ТОО в Казахстане?",
            "Какова процедура расторжения трудового договора в РК?",
            "Какие налоги обязано платить ТОО в Казахстане?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. 📝 Вопрос: {question}")
            
            prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
                
                # Extract answer
                if "<start_of_turn>model\n" in response:
                    answer = response.split("<start_of_turn>model\n")[-1]
                    if "<end_of_turn>" in answer:
                        answer = answer.split("<end_of_turn>")[0]
                else:
                    answer = response
                
                print(f"🤖 Ответ: {answer.strip()}")
                print("-" * 60)
                
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                print("-" * 60)
    
    except ImportError:
        print("❌ MLX не установлен")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    quick_test() 