#!/usr/bin/env python3
"""
System check script for Gemma fine-tuning on Mac
"""

import sys
import platform
import subprocess
import json

def check_system():
    """Check system specifications"""
    print("🖥️  ПРОВЕРКА СИСТЕМЫ")
    print("=" * 40)
    
    # Basic system info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Check if it's Apple Silicon
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':
        print("✅ Apple Silicon обнаружен")
    else:
        print("⚠️  Не Apple Silicon - производительность может быть ниже")

def check_memory():
    """Check memory"""
    print("\n💾 ПРОВЕРКА ПАМЯТИ")
    print("=" * 40)
    
    try:
        # Get memory info on macOS
        result = subprocess.run(['sysctl', 'hw.memsize'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            mem_bytes = int(result.stdout.split()[-1])
            mem_gb = mem_bytes / (1024**3)
            print(f"Общая память: {mem_gb:.1f} GB")
            
            if mem_gb >= 16:
                print("✅ Достаточно памяти для обучения")
            elif mem_gb >= 8:
                print("⚠️  Минимальная память - используйте маленькие батчи")
            else:
                print("❌ Недостаточно памяти для комфортного обучения")
        else:
            print("❓ Не удалось определить объем памяти")
    except:
        print("❓ Не удалось проверить память")

def check_pytorch():
    """Check PyTorch and MPS"""
    print("\n🔥 ПРОВЕРКА PYTORCH И MPS")
    print("=" * 40)
    
    try:
        import torch
        print(f"PyTorch версия: {torch.__version__}")
        
        # Check MPS
        if torch.backends.mps.is_available():
            print("✅ MPS доступен для ускорения")
            if torch.backends.mps.is_built():
                print("✅ MPS собран корректно")
            else:
                print("⚠️  MPS может работать нестабильно")
        else:
            print("❌ MPS недоступен - будет использоваться CPU")
            
        # Memory test
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            x = torch.randn(1000, 1000).to(device)
            y = torch.mm(x, x.t())
            print(f"✅ Тест операций на {device} прошел успешно")
        except Exception as e:
            print(f"❌ Ошибка теста: {e}")
            
    except ImportError:
        print("❌ PyTorch не установлен")

def check_dependencies():
    """Check required packages"""
    print("\n📦 ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    print("=" * 40)
    
    required_packages = [
        'torch',
        'transformers', 
        'datasets',
        'peft',
        'accelerate',
        'tokenizers',
        'huggingface_hub'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - отсутствует")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Установите недостающие пакеты:")
        print(f"pip install {' '.join(missing)}")

def check_dataset():
    """Check if dataset exists"""
    print("\n📊 ПРОВЕРКА ДАТАСЕТА")
    print("=" * 40)
    
    import os
    dataset_path = "final_training_dataset/kazakh_law_qa_high_quality_20250702_013138.jsonl"
    
    if os.path.exists(dataset_path):
        print("✅ Датасет найден")
        
        # Check file size
        size_mb = os.path.getsize(dataset_path) / (1024**2)
        print(f"Размер: {size_mb:.1f} MB")
        
        # Quick content check
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                data = json.loads(first_line)
                print(f"✅ Формат JSON корректен")
                
                required_fields = ['instruction', 'output']
                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    print(f"⚠️  Отсутствуют поля: {missing_fields}")
                else:
                    print("✅ Все необходимые поля присутствуют")
                    
        except Exception as e:
            print(f"❌ Ошибка чтения датасета: {e}")
    else:
        print("❌ Датасет не найден")
        print(f"Ожидаемый путь: {dataset_path}")

def check_huggingface():
    """Check Hugging Face access"""
    print("\n🤗 ПРОВЕРКА HUGGING FACE")
    print("=" * 40)
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Check if logged in
        try:
            user = api.whoami()
            print(f"✅ Вошли как: {user['name']}")
        except:
            print("⚠️  Не авторизованы в Hugging Face")
            print("Выполните: huggingface-cli login")
            
    except ImportError:
        print("❌ huggingface_hub не установлен")

def estimate_training_time():
    """Estimate training time"""
    print("\n⏰ ОЦЕНКА ВРЕМЕНИ ОБУЧЕНИЯ")
    print("=" * 40)
    
    # Read dataset size
    dataset_path = "final_training_dataset/kazakh_law_qa_high_quality_20250702_013138.jsonl"
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = sum(1 for _ in f)
        
        print(f"Количество образцов: {lines}")
        
        # Rough estimates for Mac M1 Pro
        estimates = {
            "Gemma 2B (тест, 500 образцов)": "1-2 часа",
            "Gemma 2B (полный датасет)": f"{lines//500 * 2}-{lines//500 * 3} часов", 
            "Gemma 4B (полный датасет)": f"{lines//250 * 2}-{lines//250 * 4} часов"
        }
        
        for model, time in estimates.items():
            print(f"• {model}: ~{time}")
            
    except:
        print("❓ Не удалось оценить время")

def main():
    """Run all checks"""
    print("🔍 ПРОВЕРКА ГОТОВНОСТИ СИСТЕМЫ ДЛЯ ОБУЧЕНИЯ GEMMA")
    print("=" * 60)
    
    check_system()
    check_memory()
    check_pytorch()
    check_dependencies()
    check_dataset()
    check_huggingface()
    estimate_training_time()
    
    print("\n" + "=" * 60)
    print("📋 ИТОГ")
    print("=" * 60)
    print("Если все пункты отмечены ✅, можете запускать обучение!")
    print("При наличии ⚠️  или ❌ - исправьте проблемы перед стартом")

if __name__ == "__main__":
    main() 