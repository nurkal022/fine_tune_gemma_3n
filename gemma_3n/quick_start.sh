#!/bin/bash

echo "🚀 БЫСТРЫЙ СТАРТ ОБУЧЕНИЯ GEMMA 3 НА MAC M1 PRO"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "gemma_env" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv gemma_env
fi

echo "🔄 Активация окружения..."
source gemma_env/bin/activate

echo "⬇️  Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🔍 Проверка системы..."
python check_system.py

echo ""
echo "🔑 Настройка доступа к Gemma..."
python setup_gemma_access.py

echo ""
echo "✅ ГОТОВО К ЗАПУСКУ!"
echo "==================="
echo ""
echo "Следующие шаги:"
echo "1. Получите доступ к Gemma моделям:"
echo "   https://huggingface.co/google/gemma-3-1b-it"
echo "   Нажмите 'Agree and access repository'"
echo ""
echo "2. Авторизуйтесь в HuggingFace:"
echo "   huggingface-cli login"
echo ""
echo "3. Запустите обучение:"
echo "   MLX (рекомендуется): python fine_tune_gemma_mlx.py"
echo "   PyTorch: python fine_tune_gemma_local.py"
echo ""
echo "4. Протестируйте модель:"
echo "   python test_mlx_model.py или python test_model.py"
echo ""
echo "📖 Полные инструкции в TRAINING_INSTRUCTIONS.md" 