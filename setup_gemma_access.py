#!/usr/bin/env python3
"""
Helper script to guide through Gemma access setup
"""

import webbrowser
import os
import sys

def check_hf_login():
    """Check if user is logged into HuggingFace"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✅ Вы вошли как: {user['name']}")
        return True
    except:
        print("❌ Вы не авторизованы в HuggingFace")
        return False

def check_model_access():
    """Check if user has access to Gemma models"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        print("✅ Доступ к google/gemma-3-1b-it получен")
        return True
    except Exception as e:
        if "gated repo" in str(e).lower():
            print("❌ Нет доступа к google/gemma-3-1b-it")
            return False
        else:
            print(f"❓ Ошибка проверки: {e}")
            return False

def open_gemma_page():
    """Open Gemma model page in browser"""
    print("🌐 Открываю страницу Gemma в браузере...")
    webbrowser.open("https://huggingface.co/google/gemma-3-1b-it")

def login_to_hf():
    """Guide user through HF login"""
    print("\n🔑 Процесс авторизации в HuggingFace:")
    print("1. Перейдите на https://huggingface.co/settings/tokens")
    print("2. Создайте новый токен с правами чтения (Read)")
    print("3. Скопируйте токен")
    print("4. Вставьте его в команду ниже")
    print("\nВыполните:")
    print("huggingface-cli login")

def main():
    """Main setup wizard"""
    print("🏛️ НАСТРОЙКА ДОСТУПА К GEMMA 3 1B")
    print("=" * 50)
    
    print("\nПроверяю текущий статус...")
    
    # Check HF login
    hf_logged_in = check_hf_login()
    
    # Check model access
    model_access = False
    if hf_logged_in:
        model_access = check_model_access()
    
    print("\n" + "="*50)
    print("📋 ПЛАН ДЕЙСТВИЙ:")
    print("="*50)
    
    step = 1
    
    if not hf_logged_in:
        print(f"{step}. 🔑 АВТОРИЗАЦИЯ В HUGGINGFACE")
        login_to_hf()
        step += 1
        print()
    
    if not model_access:
        print(f"{step}. 📝 ПОЛУЧЕНИЕ ДОСТУПА К GEMMA")
        print("   a) Перейдите на страницу модели (откроется автоматически)")
        print("   b) Нажмите 'Agree and access repository'")
        print("   c) Дождитесь одобрения (обычно мгновенно)")
        
        try:
            open_gemma_page()
        except:
            print("   Или перейдите вручную: https://huggingface.co/google/gemma-3-1b-it")
        
        step += 1
        print()
    
    if hf_logged_in and model_access:
        print("✅ ВСЕ ГОТОВО! Можете запускать обучение:")
        print()
        print("   MLX (рекомендуется):")
        print("   python fine_tune_gemma_mlx.py")
        print()
        print("   PyTorch:")
        print("   python fine_tune_gemma_local.py")
        return
    
    print(f"{step}. ✅ ПРОВЕРКА")
    print("   После выполнения предыдущих шагов запустите:")
    print("   python setup_gemma_access.py")
    print()
    
    print("💡 ПОЛЕЗНЫЕ ССЫЛКИ:")
    print("   📚 Документация: README.md")
    print("   🛠️  Инструкции: TRAINING_INSTRUCTIONS.md")
    print("   🤗 HuggingFace Tokens: https://huggingface.co/settings/tokens")
    print("   🏛️  Gemma 3 1B: https://huggingface.co/google/gemma-3-1b-it")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("Попробуйте запустить скрипт еще раз") 