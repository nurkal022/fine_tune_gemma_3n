#!/usr/bin/env python3
"""
Базовый пример использования модели Gemma 3 4B для казахского права
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma_legal_model import KazakhLegalModel

def main():
    """Основная функция для демонстрации базового использования"""
    
    print("🇰🇿 Загружаем модель казахского правового эксперта...")
    
    # Инициализация модели
    model = KazakhLegalModel()
    
    # Примеры вопросов
    questions = [
        "ТОО құру үшін қандай құжаттар қажет?",
        "Еңбекшіні жұмыстан шығару үшін қандай негіздер бар?",
        "Жеке кәсіпкер ретінде тіркелу үшін қандай шарттар бар?",
        "Қандай жағдайда жұмыс беруші еңбекшіге айыппұл сала алады?",
        "Меншік құқығы қалай тіркеледі?"
    ]
    
    print("\n" + "="*60)
    print("📋 ДЕМОНСТРАЦИЯ РАБОТЫ МОДЕЛИ")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Вопрос {i}: {question}")
        print("-" * 50)
        
        try:
            # Получение ответа от модели
            answer = model.ask(question)
            print(f"🤖 Ответ: {answer}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print("-" * 50)
    
    print("\n✅ Демонстрация завершена!")

if __name__ == "__main__":
    main()
