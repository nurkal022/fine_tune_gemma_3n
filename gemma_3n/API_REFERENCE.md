# 🔌 API Справочник Gemma 3 4B

## 📚 Основные функции

### load() - Загрузка модели
```python
from mlx_lm import load

model, tokenizer = load(
    path="google/gemma-3-4b-it",           # Базовая модель
    adapter_path="./checkpoints_full",      # Путь к адаптерам
    tokenizer_config={}                     # Дополнительные настройки
)
```

**Параметры:**
- `path`: Путь к базовой модели (строка)
- `adapter_path`: Путь к обученным адаптерам (строка)
- `tokenizer_config`: Конфигурация токенизатора (словарь)

**Возвращает:**
- `model`: Загруженная модель
- `tokenizer`: Токенизатор

### generate() - Генерация текста
```python
from mlx_lm import generate

response = generate(
    model=model,                    # Загруженная модель
    tokenizer=tokenizer,           # Токенизатор
    prompt="Ваш промпт",           # Входной текст
    max_tokens=200,                # Макс. длина ответа
    temp=0.1,                      # Температура генерации
    top_p=0.9,                     # Nucleus sampling
    repetition_penalty=1.0,        # Штраф за повторы
    stream=False,                  # Потоковая генерация
    verbose=False                  # Детальный вывод
)
```

**Параметры:**
- `model`: Модель для генерации (обязательный)
- `tokenizer`: Токенизатор (обязательный)  
- `prompt`: Входной промпт (строка)
- `max_tokens`: Максимальное количество токенов (int, по умолчанию: 100)
- `temp`: Температура для случайности (float, 0.0-2.0, по умолчанию: 0.0)
- `top_p`: Nucleus sampling параметр (float, 0.0-1.0, по умолчанию: 1.0)
- `repetition_penalty`: Штраф за повторы (float, по умолчанию: 1.0)
- `stream`: Потоковая генерация (bool, по умолчанию: False)
- `verbose`: Подробный вывод (bool, по умолчанию: False)

**Возвращает:**
- `str`: Сгенерированный текст

## 🎛️ Форматы промптов

### Базовый формат Gemma
```python
prompt = f"""<bos><start_of_turn>user
{ваш_вопрос}<end_of_turn>
<start_of_turn>model
"""
```

### Контекстуальный промпт
```python
def format_legal_prompt(question, context=None):
    if context:
        prompt = f"""<bos><start_of_turn>user
Контекст: {context}

Вопрос: {question}<end_of_turn>
<start_of_turn>model
"""
    else:
        prompt = f"""<bos><start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""
    return prompt
```

### Пошаговый анализ
```python
def format_step_by_step_prompt(question):
    prompt = f"""<bos><start_of_turn>user
Объясни пошагово: {question}

Требуется:
1. Краткое определение
2. Пошаговая процедура  
3. Практические примеры
4. Ссылки на законы<end_of_turn>
<start_of_turn>model
"""
    return prompt
```

## 🔧 Вспомогательные функции

### ask_legal_question() - Упрощенный интерфейс
```python
def ask_legal_question(question, model, tokenizer, max_tokens=300, temp=0.1):
    """
    Задать вопрос по казахскому праву
    
    Args:
        question (str): Правовой вопрос
        model: Загруженная модель
        tokenizer: Токенизатор
        max_tokens (int): Максимальная длина ответа
        temp (float): Температура генерации
        
    Returns:
        str: Ответ модели
    """
    prompt = format_legal_prompt(question)
    
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp
    )
    
    # Очистка ответа
    if "<end_of_turn>" in response:
        response = response.split("<end_of_turn>")[0]
    
    return response.strip()
```

### batch_process() - Пакетная обработка
```python
def batch_process(questions, model, tokenizer, max_tokens=200):
    """
    Обработка списка вопросов
    
    Args:
        questions (list): Список вопросов
        model: Загруженная модель
        tokenizer: Токенизатор
        max_tokens (int): Макс. длина каждого ответа
        
    Returns:
        list: Список ответов
    """
    results = []
    for i, question in enumerate(questions):
        print(f"Обработка {i+1}/{len(questions)}: {question[:50]}...")
        
        answer = ask_legal_question(
            question, model, tokenizer, max_tokens
        )
        
        results.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
    
    return results
```

### save_conversation() - Сохранение диалога
```python
import json
from datetime import datetime

def save_conversation(conversation, filename=None):
    """
    Сохранение диалога в JSON файл
    
    Args:
        conversation (list): Список сообщений
        filename (str): Имя файла (опционально)
        
    Returns:
        str: Путь к сохраненному файлу
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "model": "gemma-3-4b-legal-kz",
        "conversation": conversation
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filename
```

## 📊 Мониторинг и метрики

### performance_monitor() - Мониторинг производительности
```python
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Начать мониторинг"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
    def stop(self):
        """Закончить мониторинг и вернуть метрики"""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            "duration": end_time - self.start_time,
            "memory_start": self.start_memory,
            "memory_end": end_memory,
            "memory_peak": end_memory - self.start_memory
        }

# Использование
monitor = PerformanceMonitor()
monitor.start()
response = ask_legal_question("Что такое ТОО?", model, tokenizer)
metrics = monitor.stop()

print(f"Время: {metrics['duration']:.2f} сек")
print(f"Память: {metrics['memory_peak']:.1f} MB")
```

### calculate_tokens() - Подсчет токенов
```python
def calculate_tokens(text, tokenizer):
    """
    Подсчет количества токенов в тексте
    
    Args:
        text (str): Входной текст
        tokenizer: Токенизатор
        
    Returns:
        int: Количество токенов
    """
    tokens = tokenizer.encode(text)
    return len(tokens)

def estimate_cost(prompt, max_tokens, tokenizer):
    """
    Оценка "стоимости" генерации
    
    Args:
        prompt (str): Входной промпт
        max_tokens (int): Макс. токенов ответа
        tokenizer: Токенизатор
        
    Returns:
        dict: Оценки времени и ресурсов
    """
    input_tokens = calculate_tokens(prompt, tokenizer)
    total_tokens = input_tokens + max_tokens
    
    # Оценки на основе бенчмарков
    estimated_time = total_tokens / 7.3  # 7.3 токен/сек
    estimated_memory = 9 + (total_tokens * 0.001)  # ~1MB на 1000 токенов
    
    return {
        "input_tokens": input_tokens,
        "max_output_tokens": max_tokens,
        "total_tokens": total_tokens,
        "estimated_time": estimated_time,
        "estimated_memory_mb": estimated_memory
    }
```

## 🔄 Потоковая генерация

### stream_generate() - Потоковая генерация
```python
def stream_generate(question, model, tokenizer, max_tokens=300):
    """
    Потоковая генерация с выводом в реальном времени
    
    Args:
        question (str): Вопрос
        model: Модель
        tokenizer: Токенизатор
        max_tokens (int): Макс. токенов
        
    Yields:
        str: Части ответа
    """
    prompt = format_legal_prompt(question)
    
    # Потоковая генерация (если поддерживается)
    for token in generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=0.1,
        stream=True
    ):
        yield token

# Использование
print("Ответ:", end=" ", flush=True)
for token in stream_generate("Что такое ТОО?", model, tokenizer):
    print(token, end="", flush=True)
print()  # Новая строка в конце
```

## 🎯 Специализированные функции

### legal_definitions() - Юридические определения
```python
def get_legal_definition(term, model, tokenizer):
    """Получить определение юридического термина"""
    prompt = f"""<bos><start_of_turn>user
Дай точное определение термина: {term}

Требуется:
- Краткое определение (1-2 предложения)
- Источник в казахстанском праве
- Практическое применение<end_of_turn>
<start_of_turn>model
"""
    
    return generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=200,
        temp=0.05  # Очень низкая температура для точности
    )
```

### legal_procedure() - Правовые процедуры
```python
def get_legal_procedure(procedure, model, tokenizer):
    """Получить описание правовой процедуры"""
    prompt = f"""<bos><start_of_turn>user
Опиши пошагово процедуру: {procedure}

Нужно указать:
1. Необходимые документы
2. Пошаговый алгоритм
3. Сроки выполнения
4. Возможные сложности<end_of_turn>
<start_of_turn>model
"""
    
    return generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=400,
        temp=0.1
    )
```

## 📱 Интеграция с веб-фреймворками

### Flask API
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Глобальная загрузка модели (один раз при старте)
model, tokenizer = load(
    "google/gemma-3-4b-it",
    adapter_path="./checkpoints_full"
)

@app.route('/ask', methods=['POST'])
def api_ask():
    data = request.json
    question = data.get('question')
    max_tokens = data.get('max_tokens', 200)
    
    if not question:
        return jsonify({'error': 'Вопрос не указан'}), 400
    
    try:
        answer = ask_legal_question(question, model, tokenizer, max_tokens)
        return jsonify({
            'question': question,
            'answer': answer,
            'model': 'gemma-3-4b-legal-kz'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI (асинхронный)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Gemma Legal API", version="1.0.0")

class QuestionRequest(BaseModel):
    question: str
    max_tokens: int = 200
    temperature: float = 0.1

class AnswerResponse(BaseModel):
    question: str
    answer: str
    model: str = "gemma-3-4b-legal-kz"

# Загрузка модели при старте
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model, tokenizer = load(
        "google/gemma-3-4b-it",
        adapter_path="./checkpoints_full"
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        answer = ask_legal_question(
            request.question, 
            model, 
            tokenizer, 
            request.max_tokens,
            request.temperature
        )
        return AnswerResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

**📖 Совет:** Всегда используйте низкую температуру (0.05-0.1) для правовых вопросов! 