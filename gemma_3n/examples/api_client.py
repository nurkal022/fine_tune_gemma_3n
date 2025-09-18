#!/usr/bin/env python3
"""
Пример клиента для API сервера казахского правового эксперта
"""

import requests
import json
import time

class LegalAPIClient:
    """Клиент для работы с API правового эксперта"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def ask_question(self, question, max_tokens=512, temperature=0.7):
        """Задать вопрос модели через API"""
        
        payload = {
            "question": question,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/ask",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": f"API Error: {e}"}
    
    def get_health(self):
        """Проверить статус API"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Health check failed: {e}"}

def main():
    """Демонстрация работы с API"""
    
    print("🌐 Подключение к API серверу...")
    
    client = LegalAPIClient()
    
    # Проверка здоровья API
    health = client.get_health()
    if "error" in health:
        print(f"❌ Не удалось подключиться к API: {health['error']}")
        print("💡 Убедитесь, что сервер запущен: python api_server.py")
        return
    
    print(f"✅ API сервер работает: {health}")
    
    # Примеры вопросов
    questions = [
        "ТОО құру үшін қандай құжаттар қажет?",
        "Еңбекшіні жұмыстан шығару үшін қандай негіздер бар?",
        "Жеке кәсіпкер ретінде тіркелу үшін қандай шарттар бар?"
    ]
    
    print("\n" + "="*60)
    print("📋 ДЕМОНСТРАЦИЯ API КЛИЕНТА")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Вопрос {i}: {question}")
        print("-" * 50)
        
        start_time = time.time()
        result = client.ask_question(question)
        end_time = time.time()
        
        if "error" in result:
            print(f"❌ Ошибка: {result['error']}")
        else:
            print(f"🤖 Ответ: {result.get('answer', 'Нет ответа')}")
            print(f"⏱️ Время ответа: {end_time - start_time:.2f} сек")
            print(f"📊 Токенов: {result.get('tokens_used', 'N/A')}")
        
        print("-" * 50)
    
    print("\n✅ Демонстрация API завершена!")

if __name__ == "__main__":
    main()
