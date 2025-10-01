# Казахстанский правовой QA датасет для обучения модели

## 📊 Статистика датасета
- **Всего записей**: 8,652
- **Высококачественных записей**: 3,753 (качество ≥ 0.7)
- **Средняя оценка качества**: 0.676
- **Общий объем**: 8,110,425 символов, 922,821 слов

## 📁 Файлы датасета

### Для обучения модели:
1. **`kazakh_law_qa_simple_20250702_013138.jsonl`** - Простой формат (instruction, input, output)
2. **`kazakh_law_qa_high_quality_20250702_013138.jsonl`** - Только высококачественные записи
3. **`kazakh_law_qa_full_20250702_013138.jsonl`** - Полный датасет с метаданными

### Для анализа:
4. **`kazakh_law_qa_analysis_20250702_013138.csv`** - CSV для анализа
5. **`dataset_stats_20250702_013138.json`** - Детальная статистика

## 🎯 Распределение по сложности
- **medium**: 8,564 записей
- **complex**: 88 записей

## 🚀 Использование

### Для instruction tuning:
```python
import json

# Загрузка простого формата
with open('kazakh_law_qa_simple_20250702_013138.jsonl', 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Каждая запись содержит:
# - instruction: вопрос на казахском/русском языке
# - input: пустая строка (для совместимости)
# - output: ответ по казахстанскому праву
```

### Для анализа:
```python
import pandas as pd

# Загрузка CSV для анализа
df = pd.read_csv('kazakh_law_qa_analysis_20250702_013138.csv')
print(df.groupby('complexity')['quality_score'].describe())
```

## 📋 Формат записи

```json
{
  "instruction": "Вопрос по казахстанскому праву",
  "input": "",
  "output": "Подробный ответ со ссылками на законодательство",
  "quality_score": 0.85,
  "complexity": "medium",
  "source_file": "исходный_файл.jsonl"
}
```

Создано: 2025-07-02 01:31:38
