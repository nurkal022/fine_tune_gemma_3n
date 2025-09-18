# 🔧 Устранение проблем Gemma 3 4B

## 🚨 Частые проблемы и решения

### 1. Ошибки загрузки модели

#### Проблема: "No module named 'mlx_lm'"
```bash
ImportError: No module named 'mlx_lm'
```

**Решение:**
```bash
# Активируйте виртуальное окружение
source ../gemma_env/bin/activate

# Проверьте установку
pip list | grep mlx

# Переустановите если нужно
pip install mlx-lm
```

#### Проблема: "Checkpoint not found"
```bash
FileNotFoundError: [Errno 2] No such file or directory: './checkpoints_full'
```

**Решение:**
```bash
# Проверьте наличие чекпоинтов
ls -la checkpoints_full/

# Должны быть файлы:
# adapters.safetensors
# adapter_config.json
# 0001000_adapters.safetensors

# Если нет - запустите обучение заново
python full_training.py
```

#### Проблема: "Model loading timeout"
```bash
TimeoutError: Model loading took too long
```

**Решение:**
```bash
# Очистите кэш HuggingFace
rm -rf ~/.cache/huggingface/

# Перезапустите с увеличенным таймаутом
export HF_HUB_DISABLE_PROGRESS_BARS=1
python test_gemma_4b_full.py
```

### 2. Проблемы с памятью

#### Проблема: "Out of Memory (OOM)"
```bash
RuntimeError: [MPS] out of memory
```

**Решение:**
```bash
# Проверьте доступную память
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available/1024**3:.1f}GB')"

# Если меньше 12GB - используйте меньшую модель
cd ../  # Вернитесь к Gemma 1B
python test_gemma_1b.py

# Или уменьшите max_tokens
python test_gemma_4b_full.py benchmark --max_tokens 100
```

#### Проблема: Медленная генерация
```bash
# Скорость ниже 5 токенов/сек
```

**Решение:**
```python
# Проверьте использование ресурсов
import psutil

# CPU
print(f"CPU: {psutil.cpu_percent()}%")

# Память
mem = psutil.virtual_memory()
print(f"RAM: {mem.percent}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB)")

# Закройте другие приложения
# Уменьшите max_tokens
# Используйте temperature=0.1 (быстрее)
```

### 3. Проблемы качества ответов

#### Проблема: Повторяющийся текст
```
Ответ: "ТОО это ТОО это ТОО это..."
```

**Решение:**
```python
# Увеличьте repetition_penalty
response = generate(
    model, tokenizer,
    prompt=prompt,
    repetition_penalty=1.1,  # Или 1.2
    temp=0.1
)
```

#### Проблема: Слишком короткие ответы
```
Ответ: "ТОО - это организация."
```

**Решение:**
```python
# Увеличьте max_tokens и улучшите промпт
prompt = f"""<bos><start_of_turn>user
Подробно объясни: {question}

Требуется:
- Детальное определение
- Практические примеры
- Ссылки на законы<end_of_turn>
<start_of_turn>model
"""

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=400,  # Больше токенов
    temp=0.1
)
```

#### Проблема: Неточные ответы
```
Ответ содержит информацию не по Казахстану
```

**Решение:**
```python
# Укажите юрисдикцию в промпте
prompt = f"""<bos><start_of_turn>user
Согласно законодательству Республики Казахстан: {question}<end_of_turn>
<start_of_turn>model
"""

# Используйте очень низкую температуру
response = generate(
    model, tokenizer,
    prompt=prompt,
    temp=0.05,  # Максимальная точность
    max_tokens=300
)
```

### 4. Проблемы производительности

#### Проблема: Долгая загрузка модели
```bash
# Загрузка занимает больше 10 секунд
```

**Решение:**
```python
# Предзагрузите модель один раз
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load_once(self):
        if self.model is None:
            print("Загрузка модели...")
            self.model, self.tokenizer = load(
                "google/gemma-3-4b-it",
                adapter_path="./checkpoints_full"
            )
        return self.model, self.tokenizer

# Глобальный менеджер
manager = ModelManager()
model, tokenizer = manager.load_once()
```

#### Проблема: Высокое использование CPU
```bash
# CPU загружен на 100%
```

**Решение:**
```bash
# Ограничьте количество потоков
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Или в Python
import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# Закройте браузер и другие приложения
```

### 5. Проблемы с JSON и форматированием

#### Проблема: JSON parsing errors
```bash
JSONDecodeError: Expecting ',' delimiter
```

**Решение:**
```bash
# Проверьте формат данных
python fix_dataset.py

# Исправьте кодировку
python -c "
import json
with open('data/train.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except Exception as e:
            print(f'Ошибка в строке {i+1}: {e}')
"
```

#### Проблема: Неправильные символы в ответах
```
Ответ: "ÐÐÐ â ÐÐ°Ð·Ð°ÑÑÑÐµ"
```

**Решение:**
```python
# Принудительно используйте UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8')

# При сохранении
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(response)

# При загрузке
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

### 6. Системные проблемы

#### Проблема: "MPS не доступен"
```bash
RuntimeError: MPS backend is not available
```

**Решение:**
```bash
# Проверьте версию macOS (нужна 12.3+)
sw_vers

# Проверьте PyTorch
python -c "import torch; print(torch.backends.mps.is_available())"

# Используйте CPU если MPS недоступен
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### Проблема: Медленный SSD
```bash
# Модель загружается очень медленно
```

**Решение:**
```bash
# Проверьте свободное место
df -h

# Очистите кэш если нужно
rm -rf ~/.cache/huggingface/transformers/

# Проверьте скорость диска
time dd if=/dev/zero of=test_file bs=1M count=1000
rm test_file
```

## 🔍 Диагностические команды

### Системная информация
```bash
# Проверка системы
python check_system.py

# Информация о памяти
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Общая RAM: {mem.total/1024**3:.1f}GB')
print(f'Доступно: {mem.available/1024**3:.1f}GB')
print(f'Использовано: {mem.percent}%')
"

# Информация о MLX
python -c "
import mlx.core as mx
print(f'MLX версия: {mx.__version__}')
print(f'Unified Memory: {mx.metal.get_peak_memory()/1024**3:.1f}GB')
"
```

### Тест модели
```python
# Быстрый тест
def quick_test():
    try:
        from mlx_lm import load, generate
        print("✅ MLX импортирован")
        
        model, tokenizer = load(
            "google/gemma-3-4b-it",
            adapter_path="./checkpoints_full"
        )
        print("✅ Модель загружена")
        
        response = generate(
            model, tokenizer,
            prompt="<bos><start_of_turn>user\nТест<end_of_turn>\n<start_of_turn>model\n",
            max_tokens=10
        )
        print(f"✅ Генерация работает: {response}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

quick_test()
```

## 📞 Получение помощи

### Логи для отладки
```bash
# Включите подробные логи
export PYTHONPATH="/path/to/project:$PYTHONPATH"
export MLX_LOG_LEVEL=DEBUG
python test_gemma_4b_full.py benchmark > debug.log 2>&1

# Отправьте debug.log для анализа
```

### Сбор диагностической информации
```python
import platform
import sys
import mlx.core as mx
from mlx_lm import __version__ as mlx_lm_version

print("=== ДИАГНОСТИЧЕСКАЯ ИНФОРМАЦИЯ ===")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"MLX: {mx.__version__}")
print(f"MLX-LM: {mlx_lm_version}")
print(f"Архитектура: {platform.machine()}")

# Проверка памяти
import psutil
mem = psutil.virtual_memory()
print(f"RAM: {mem.total/1024**3:.1f}GB (доступно: {mem.available/1024**3:.1f}GB)")

# Проверка чекпоинтов
import os
if os.path.exists("checkpoints_full"):
    files = os.listdir("checkpoints_full")
    print(f"Чекпоинты: {files}")
else:
    print("❌ Папка checkpoints_full не найдена")
```

## 🆘 Критические ошибки

### Если ничего не работает
```bash
# 1. Полная переустановка
rm -rf ../gemma_env
python -m venv ../gemma_env
source ../gemma_env/bin/activate
pip install mlx-lm transformers datasets

# 2. Заново скачайте чекпоинты
rm -rf checkpoints_full
python full_training.py

# 3. Проверьте, что M1/M2 Mac
system_profiler SPHardwareDataType | grep "Chip"
```

### Откат к рабочей версии
```bash
# Используйте простую Gemma 1B
cd ../
python test_gemma_1b.py

# Или минимальный тест
python -c "
from mlx_lm import load, generate
model, tokenizer = load('mlx-community/gemma-2-2b-it-4bit')
print(generate(model, tokenizer, prompt='Hello', max_tokens=5))
"
```

---

**📞 Если проблема не решается:** Создайте issue с диагностической информацией и логами. 