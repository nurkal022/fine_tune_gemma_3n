# Инструкции по запуску обучения Gemma 3 на Mac M1 Pro

## 1. Подготовка окружения

### Создание виртуального окружения:
```bash
python3 -m venv gemma_env
source gemma_env/bin/activate
```

### Установка зависимостей:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Настройка Hugging Face и доступ к Gemma:
```bash
pip install huggingface_hub
huggingface-cli login
# Введите ваш HF токен с доступом на чтение
```

**⚠️ ОБЯЗАТЕЛЬНО:** Получите доступ к Gemma моделям:
1. Перейдите на https://huggingface.co/google/gemma-3-1b-it
2. Нажмите "Agree and access repository" 
3. Дождитесь одобрения (обычно мгновенно)
4. Повторите для других версий Gemma если нужно

## 2. Выбор метода обучения

### Вариант A: MLX (рекомендуется для Mac)
**Преимущества:** Быстрее в 2-3 раза, оптимизирован для Apple Silicon

### Вариант B: PyTorch с PEFT
**Преимущества:** Стандартный подход, больше настроек

## 3. Запуск обучения

### MLX (рекомендуется):
```bash
python fine_tune_gemma_mlx.py  # Автоматически установит MLX
python test_mlx_model.py       # Тестирование
```

### PyTorch:
```bash
python fine_tune_gemma_local.py  # Тестовый запуск (500 образцов)
python test_model.py             # Тестирование
```

### Полный датасет:
Измените в скрипте:
```python
train_data = raw_data[:500]  # На:
train_data = raw_data  # Весь датасет (3,753 образца)
```

## 4. Ожидаемое время и ресурсы

### MLX (M1 Pro 16GB):
- **500 образцов**: 30-60 минут
- **Полный датасет (3,753)**: 3-5 часов  
- **Память**: 4-6GB

### PyTorch (M1 Pro 16GB):
- **500 образцов**: 1-2 часа
- **Полный датасет (3,753)**: 6-10 часов
- **Память**: 8-12GB

## 5. Мониторинг процесса

Скрипт выводит:
- Количество обучаемых параметров
- Логи каждые 10 шагов
- Сохранение каждые 100 шагов

## 6. Результат

После обучения:
- Модель сохраняется в `./gemma_kazakh_law_finetuned/`
- Автоматический тест на примере вопроса
- Файлы готовы для inference

## 7. Использование обученной модели

### MLX:
```python
from mlx_lm import load, generate

# Загрузка
model, tokenizer = load("google/gemma-3-1b-it", adapter_path="./checkpoints")

# Inference
question = "Ваш вопрос по казахскому праву"
prompt = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
answer = generate(model, tokenizer, prompt=prompt, max_tokens=256)
```

### PyTorch:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Загрузка
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
model = PeftModel.from_pretrained(base_model, "./gemma_kazakh_law_finetuned")
tokenizer = AutoTokenizer.from_pretrained("./gemma_kazakh_law_finetuned")

# Inference
question = "Ваш вопрос по казахскому праву"
inputs = tokenizer(f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 8. Troubleshooting

### Если не хватает памяти:
- Уменьшите `per_device_train_batch_size` до 1
- Увеличьте `gradient_accumulation_steps` до 8
- Уменьшите `max_length` до 256

### Если MPS не работает:
- Скрипт автоматически переключится на CPU
- Время обучения увеличится в 3-5 раз

### Если ошибки с токенизатором:
```bash
# Очистите кеш
rm -rf ~/.cache/huggingface/
huggingface-cli login --add-to-git-credential
```

## 9. Следующие шаги

После успешного обучения:
1. Протестируйте на своих вопросах
2. Настройте параметры для лучшего качества
3. Экспериментируйте с разными LoRA конфигурациями
4. Рассмотрите квантизацию для production 