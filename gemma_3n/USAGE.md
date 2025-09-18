# 📖 Руководство по использованию Gemma 3 4B

## 🚀 Быстрый старт

### 1. Активация окружения
```bash
cd gemma_3n
source ../gemma_env/bin/activate
```

### 2. Интерактивный чат
```bash
python test_gemma_4b_full.py
```

## 💬 Примеры запросов

### Корпоративное право
```
Вопрос: Какие документы нужны для регистрации ТОО в Казахстане?
Ответ: Модель предоставит детальный список документов, сроки и процедуры.
```

### Трудовое право
```
Вопрос: Какой максимальный размер алиментов в Казахстане?
Ответ: Подробное объяснение с ссылками на законодательство.
```

### Административное право
```
Вопрос: Какие штрафы за превышение скорости?
Ответ: Актуальные размеры штрафов по категориям нарушений.
```

## 🔧 Продвинутое использование

### Программный доступ
```python
import mlx.core as mx
from mlx_lm import load, generate

# Загрузка модели
model, tokenizer = load(
    "google/gemma-3-4b-it",
    adapter_path="./checkpoints_full"
)

# Функция для вопросов
def ask_legal_question(question):
    prompt = f"""<bos><start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""
    
    response = generate(
        model, tokenizer, 
        prompt=prompt,
        max_tokens=300,
        temp=0.1
    )
    return response

# Пример использования
answer = ask_legal_question("Как расторгнуть трудовой договор?")
print(answer)
```

### Батч обработка
```python
import json

# Список вопросов
questions = [
    "Что такое ТОО?",
    "Как подать в суд?",
    "Размер минимальной зарплаты?"
]

# Обработка всех вопросов
results = []
for q in questions:
    answer = ask_legal_question(q)
    results.append({"question": q, "answer": answer})

# Сохранение результатов
with open("legal_qa_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

## 🎛️ Настройки генерации

### Параметры качества
```python
# Точные ответы (рекомендуется для права)
response = generate(
    model, tokenizer, prompt=prompt,
    temp=0.1,           # Низкая температура = точность
    max_tokens=500,     # Длинные объяснения
    repetition_penalty=1.1
)

# Креативные ответы
response = generate(
    model, tokenizer, prompt=prompt,
    temp=0.7,           # Высокая температура = креативность
    max_tokens=300,
    top_p=0.9
)
```

### Форматы промптов

#### Базовый вопрос
```
<bos><start_of_turn>user
{ваш_вопрос}<end_of_turn>
<start_of_turn>model
```

#### С контекстом
```
<bos><start_of_turn>user
Контекст: Гражданин хочет открыть бизнес
Вопрос: {ваш_вопрос}<end_of_turn>
<start_of_turn>model
```

#### Пошаговое объяснение
```
<bos><start_of_turn>user
Объясни пошагово: {ваш_вопрос}<end_of_turn>
<start_of_turn>model
```

## 📊 Мониторинг производительности

### Проверка использования ресурсов
```python
import psutil
import time

def monitor_generation():
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Ваша генерация
    response = generate(model, tokenizer, prompt=prompt)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    print(f"Время: {end_time - start_time:.2f} сек")
    print(f"Память: {end_memory:.0f} MB (пик: +{end_memory-start_memory:.0f} MB)")
    
    return response
```

## 🔍 Отладка и логирование

### Включение детального логирования
```python
import logging

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ask_with_logging(question):
    logger.info(f"Задан вопрос: {question}")
    
    start_time = time.time()
    response = ask_legal_question(question)
    generation_time = time.time() - start_time
    
    logger.info(f"Ответ получен за {generation_time:.2f} сек")
    logger.info(f"Длина ответа: {len(response)} символов")
    
    return response
```

## 🎯 Лучшие практики

### ✅ Рекомендуется
- Используйте низкую температуру (0.1-0.3) для точных правовых ответов
- Задавайте конкретные вопросы
- Проверяйте ответы с первоисточниками
- Указывайте контекст для сложных ситуаций

### ❌ Избегайте
- Высокой температуры для правовых вопросов
- Слишком общих формулировок
- Вопросов без контекста для сложных случаев
- Превышения лимита токенов (замедляет генерацию)

## 🔧 Интеграция с внешними системами

### Flask API
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model, tokenizer = load("google/gemma-3-4b-it", adapter_path="./checkpoints_full")

@app.route('/ask', methods=['POST'])
def ask_legal():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'Вопрос не указан'}), 400
    
    answer = ask_legal_question(question)
    return jsonify({'question': question, 'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Telegram бот
```python
import telebot

bot = telebot.TeleBot('YOUR_BOT_TOKEN')

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    question = message.text
    answer = ask_legal_question(question)
    bot.reply_to(message, answer)

bot.polling()
```

## 📱 Команды в интерактивном режиме

| Команда | Описание |
|---------|----------|
| `exit` | Выход из чата |
| `benchmark` | Запуск теста производительности |
| `help` | Справка по командам |
| `clear` | Очистка истории чата |
| `stats` | Статистика сессии |

---

**💡 Совет:** Сохраняйте интересные диалоги в файлы для анализа качества ответов! 