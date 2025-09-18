#!/usr/bin/env python3
"""
Test script for fine-tuned Kazakh Law Gemma model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_model(model_path="./gemma_kazakh_law_finetuned"):
    """Load the fine-tuned model"""
    print("Loading model...")
    
    # Determine base model from config
    base_model_name = "google/gemma-3-1b-it"  # Change if you used different base model
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_answer(model, tokenizer, question, max_tokens=300):
    """Generate answer for a given question"""
    
    # Format prompt
    prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to device if using MPS
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
        )
    
    # Decode and extract answer
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split("<start_of_turn>model\n")[-1].strip()
    
    return answer

def main():
    """Interactive testing"""
    
    if not os.path.exists("./gemma_kazakh_law_finetuned"):
        print("Error: Fine-tuned model not found!")
        print("Make sure you've run fine_tune_gemma_local.py first")
        return
    
    try:
        model, tokenizer = load_model()
        model.eval()  # Set to evaluation mode
        
        print("\n" + "="*60)
        print("🏛️  КАЗАХСКИЙ ПРАВОВОЙ АССИСТЕНТ НА ОСНОВЕ GEMMA")
        print("="*60)
        print("Задавайте вопросы по казахстанскому праву!")
        print("Напишите 'exit' для выхода\n")
        
        # Test questions
        test_questions = [
            "Какие документы необходимы для регистрации ТОО в Казахстане?",
            "Какова процедура расторжения трудового договора в РК?",
            "Какие налоги обязано платить ТОО в Казахстане?",
            "Какова ответственность за нарушение налогового законодательства?",
        ]
        
        print("Примеры вопросов:")
        for i, q in enumerate(test_questions, 1):
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
                answer = generate_answer(model, tokenizer, question)
                print(f"\n📜 Ответ:\n{answer}\n")
                print("-" * 60)
                
            except Exception as e:
                print(f"❌ Ошибка при генерации ответа: {e}")
        
        print("До свидания!")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        print("Убедитесь, что обучение завершилось успешно")

if __name__ == "__main__":
    main() 