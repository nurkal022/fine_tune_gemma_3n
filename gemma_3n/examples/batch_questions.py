#!/usr/bin/env python3
"""
Пример пакетной обработки вопросов для тестирования модели
"""

import sys
import os
import json
import time
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma_legal_model import KazakhLegalModel

class BatchProcessor:
    """Класс для пакетной обработки вопросов"""
    
    def __init__(self):
        self.model = KazakhLegalModel()
        self.results = []
    
    def process_questions(self, questions: List[str]) -> List[Dict]:
        """Обработать список вопросов"""
        
        print(f"🔄 Обрабатываем {len(questions)} вопросов...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Вопрос {i}/{len(questions)}: {question[:50]}...")
            
            start_time = time.time()
            
            try:
                answer = self.model.ask(question)
                processing_time = time.time() - start_time
                
                result = {
                    "question": question,
                    "answer": answer,
                    "processing_time": processing_time,
                    "success": True,
                    "error": None
                }
                
                print(f"✅ Обработан за {processing_time:.2f} сек")
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                result = {
                    "question": question,
                    "answer": None,
                    "processing_time": processing_time,
                    "success": False,
                    "error": str(e)
                }
                
                print(f"❌ Ошибка: {e}")
            
            self.results.append(result)
        
        return self.results
    
    def save_results(self, filename: str = "batch_results.json"):
        """Сохранить результаты в файл"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Результаты сохранены в {filename}")
    
    def print_summary(self):
        """Вывести сводку результатов"""
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful
        
        total_time = sum(r['processing_time'] for r in self.results)
        avg_time = total_time / total if total > 0 else 0
        
        print("\n" + "="*60)
        print("📊 СВОДКА РЕЗУЛЬТАТОВ")
        print("="*60)
        print(f"📝 Всего вопросов: {total}")
        print(f"✅ Успешно обработано: {successful}")
        print(f"❌ Ошибок: {failed}")
        print(f"⏱️ Общее время: {total_time:.2f} сек")
        print(f"⚡ Среднее время: {avg_time:.2f} сек/вопрос")
        print(f"📈 Успешность: {successful/total*100:.1f}%")

def main():
    """Основная функция для демонстрации пакетной обработки"""
    
    # Примеры вопросов для тестирования
    test_questions = [
        "ТОО құру үшін қандай құжаттар қажет?",
        "Еңбекшіні жұмыстан шығару үшін қандай негіздер бар?",
        "Жеке кәсіпкер ретінде тіркелу үшін қандай шарттар бар?",
        "Қандай жағдайда жұмыс беруші еңбекшіге айыппұл сала алады?",
        "Меншік құқығы қалай тіркеледі?",
        "Қандай жағдайда келісімшартты бұзуға болады?",
        "Жалақыны қандай мерзімде төлеу керек?",
        "Жұмыс уақыты қанша сағат болуы керек?",
        "Еңбек демалысы қанша күн?",
        "Жұмыс берушінің міндеттері қандай?"
    ]
    
    print("🇰🇿 Загружаем модель для пакетной обработки...")
    
    processor = BatchProcessor()
    
    # Обработка вопросов
    results = processor.process_questions(test_questions)
    
    # Сохранение результатов
    processor.save_results("examples/batch_test_results.json")
    
    # Вывод сводки
    processor.print_summary()
    
    print("\n✅ Пакетная обработка завершена!")

if __name__ == "__main__":
    main()
