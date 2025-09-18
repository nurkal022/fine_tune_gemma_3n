# 🚀 Установка и настройка

## Системные требования

### Минимальные требования
- **ОС:** macOS 12+, Ubuntu 20.04+, Windows 10+
- **RAM:** 8 GB (для Gemma 1B), 16 GB (для Gemma 3 4B)
- **Процессор:** Apple Silicon (M1/M2) или x86_64
- **Python:** 3.8+
- **Свободное место:** 10 GB

### Рекомендуемые требования
- **ОС:** macOS 13+ (для лучшей производительности MLX)
- **RAM:** 16 GB+
- **Процессор:** Apple M1 Pro/Max/Ultra или современный x86_64
- **Python:** 3.10+
- **Свободное место:** 20 GB

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/kazakh-legal-ai.git
cd kazakh-legal-ai
```

### 2. Создание виртуального окружения

```bash
# Создание виртуального окружения
python -m venv legal_ai_env

# Активация (macOS/Linux)
source legal_ai_env/bin/activate

# Активация (Windows)
legal_ai_env\Scripts\activate
```

### 3. Установка зависимостей

```bash
# Обновление pip
pip install --upgrade pip

# Установка основных зависимостей
pip install -r requirements.txt

# Установка в режиме разработки (опционально)
pip install -e .
```

### 4. Установка MLX (macOS)

```bash
# Для Apple Silicon
pip install mlx mlx-lm

# Проверка установки
python -c "import mlx; print('MLX установлен успешно')"
```

### 5. Установка MLX (Linux/Windows)

```bash
# Установка через pip
pip install mlx mlx-lm

# Или установка из исходников
git clone https://github.com/ml-explore/mlx.git
cd mlx
pip install -e .
```

## Настройка

### 1. Конфигурация модели

Создайте файл `.env` в корне проекта:

```bash
# .env
MODEL_PATH_1B=./gemma_1b/adapters
MODEL_PATH_4B=./gemma_3n/adapters
API_HOST=localhost
API_PORT=8000
LOG_LEVEL=INFO
```

### 2. Проверка установки

```bash
# Тестирование Gemma 1B
cd gemma_1b
python test_model.py

# Тестирование Gemma 3 4B
cd ../gemma_3n
python test_model.py

# Тестирование API
python api_server.py &
python test_api.py
```

## Устранение проблем

### Проблемы с MLX

**Ошибка:** `ModuleNotFoundError: No module named 'mlx'`

**Решение:**
```bash
# Переустановка MLX
pip uninstall mlx mlx-lm
pip install mlx mlx-lm
```

### Проблемы с памятью

**Ошибка:** `OutOfMemoryError`

**Решение:**
- Уменьшите batch_size в конфигурации
- Используйте Gemma 1B вместо 3 4B
- Закройте другие приложения

### Проблемы с производительностью

**Медленная работа на x86_64:**

**Решение:**
```bash
# Установка оптимизированных версий
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Docker установка (опционально)

### 1. Создание Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "gemma_3n/api_server.py"]
```

### 2. Сборка и запуск

```bash
# Сборка образа
docker build -t kazakh-legal-ai .

# Запуск контейнера
docker run -p 8000:8000 kazakh-legal-ai
```

## Проверка работоспособности

### 1. Базовый тест

```python
from gemma_3n.gemma_legal_model import KazakhLegalModel

model = KazakhLegalModel()
answer = model.ask("ТОО құру үшін қандай құжаттар қажет?")
print(answer)
```

### 2. API тест

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Еңбекшіні жұмыстан шығару үшін қандай негіздер бар?"}'
```

### 3. Бенчмарк производительности

```bash
python benchmark.py
```

## Обновление

```bash
# Обновление кода
git pull origin main

# Обновление зависимостей
pip install -r requirements.txt --upgrade

# Перезапуск сервисов
pkill -f api_server.py
python gemma_3n/api_server.py &
```

## Удаление

```bash
# Деактивация виртуального окружения
deactivate

# Удаление виртуального окружения
rm -rf legal_ai_env

# Удаление проекта
rm -rf kazakh-legal-ai
```
