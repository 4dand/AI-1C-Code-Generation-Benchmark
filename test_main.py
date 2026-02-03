"""
Скрипт для тестирования детерминизма ИИ-моделей и проверки MCP tool calling.
Целевые модели: Claude Opus 4.5, GPT-5.2-Codex, Gemini 3 Pro
"""

import os
import json
import requests
import hashlib
import time
from datetime import datetime

# ============== КОНФИГУРАЦИЯ ==============
API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-your-api-key-here")
BASE_URL = "https://openrouter.ai/api/v1"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://1c-benchmark.local",
    "X-Title": "1C Benchmark Determinism Test"
}

# Целевые модели (актуальные ID из OpenRouter)
MODELS = {
    "opus": {
        "id": "anthropic/claude-opus-4.5",
        "name": "Claude Opus 4.5",
        "api_type": "chat",  # OpenRouter унифицирует API
        "determinism_param": "temperature",  # Для Anthropic используем temperature=0
    },
    "gpt": {
        "id": "openai/gpt-5.2-codex",
        "name": "GPT-5.2-Codex",
        "api_type": "chat",
        "determinism_param": "seed",
    },
    "gemini": {
        "id": "google/gemini-3-flash-preview",
        "name": "Gemini 3 Flash",
        "api_type": "chat",
        "determinism_param": "seed",
    }
}

# Простые задачи на 1С (минимум токенов)
TASKS_1C = [
    {
        "id": 1,
        "name": "Простая функция сложения",
        "prompt": "Напиши функцию на языке 1С:Предприятие 8 которая складывает два числа. Только код, без пояснений."
    },
    {
        "id": 2,
        "name": "Проверка на пустую строку",
        "prompt": "Напиши функцию на языке 1С:Предприятие 8 которая проверяет пустая ли строка. Возвращает Истина/Ложь. Только код."
    }
]

# MCP Tool для тестирования (русское описание работает с system prompt!)
MCP_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_1c_code",
        "description": "Выполняет код 1С:Предприятие 8 и возвращает результат",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Код на языке 1С для выполнения"
                },
                "params": {
                    "type": "object",
                    "description": "Параметры для передачи в код"
                }
            },
            "required": ["code"]
        }
    }
}


def log(message, level="INFO"):
    """Простое логирование"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def get_balance():
    """Получает текущий баланс"""
    try:
        response = requests.get(f"{BASE_URL}/auth/key", headers=HEADERS)
        if response.status_code == 200:
            data = response.json().get('data', {})
            limit = data.get('limit', 0)
            usage = data.get('usage', 0)
            return {
                "limit": limit,
                "usage": usage,
                "available": limit - usage if limit else "unlimited"
            }
    except Exception as e:
        log(f"Ошибка получения баланса: {e}", "ERROR")
    return None


def get_model_info(model_id):
    """Получает полную информацию о модели"""
    try:
        response = requests.get(f"{BASE_URL}/models", headers=HEADERS)
        if response.status_code == 200:
            models = response.json().get('data', [])
            for model in models:
                if model.get('id') == model_id:
                    return {
                        "full_id": model.get('id'),
                        "name": model.get('name'),
                        "context_length": model.get('context_length'),
                        "supported_parameters": model.get('supported_parameters', []),
                        "pricing": model.get('pricing', {}),
                        "architecture": model.get('architecture', {})
                    }
    except Exception as e:
        log(f"Ошибка получения info о модели: {e}", "ERROR")
    return None


def compute_hash(text):
    """Вычисляет MD5 хеш текста"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def call_model(model_key, prompt, seed=None, temperature=None, tools=None):
    """
    Вызов модели через OpenRouter
    Возвращает: (response_text, usage_info, elapsed_time, raw_response)
    """
    model = MODELS[model_key]
    
    request_body = {
        "model": model["id"],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }
    
    # Настройки детерминизма
    if model["determinism_param"] == "seed" and seed is not None:
        request_body["seed"] = seed
        request_body["temperature"] = 0  # Для детерминизма нужен и temperature=0
    elif model["determinism_param"] == "temperature":
        # Для Claude: temperature=0 для детерминизма
        request_body["temperature"] = temperature if temperature is not None else 0
    
    # Добавляем tools если нужно
    if tools:
        request_body["tools"] = tools
        request_body["tool_choice"] = "auto"
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=request_body,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Извлекаем ответ
            choices = data.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '')
                tool_calls = message.get('tool_calls', [])
            else:
                content = ""
                tool_calls = []
            
            # Информация об использовании
            usage = data.get('usage', {})
            
            return {
                "success": True,
                "content": content,
                "tool_calls": tool_calls,
                "usage": usage,
                "elapsed": elapsed,
                "model_used": data.get('model', model["id"]),
                "raw": data
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "elapsed": elapsed
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed": time.time() - start_time
        }


def test_determinism(model_key, task, seed1=42, seed2=42, seed3=999):
    """
    Тестирует детерминизм модели:
    - 2 запроса с одинаковым seed -> должны быть идентичны
    - 1 запрос с другим seed -> должен отличаться
    """
    model = MODELS[model_key]
    log(f"🧪 Тест детерминизма: {model['name']} | Задача: {task['name']}")
    
    results = []
    
    # Для Claude используем temperature вместо seed
    if model["determinism_param"] == "temperature":
        # Запрос 1 и 2 с temperature=0
        log(f"   Запрос 1 (temperature=0)...")
        r1 = call_model(model_key, task["prompt"], temperature=0)
        results.append(r1)
        
        log(f"   Запрос 2 (temperature=0)...")
        r2 = call_model(model_key, task["prompt"], temperature=0)
        results.append(r2)
        
        # Запрос 3 с temperature=0.7 (должен отличаться)
        log(f"   Запрос 3 (temperature=0.7)...")
        r3 = call_model(model_key, task["prompt"], temperature=0.7)
        results.append(r3)
    else:
        # Для GPT и Gemini используем seed
        log(f"   Запрос 1 (seed={seed1})...")
        r1 = call_model(model_key, task["prompt"], seed=seed1)
        results.append(r1)
        
        log(f"   Запрос 2 (seed={seed2})...")
        r2 = call_model(model_key, task["prompt"], seed=seed2)
        results.append(r2)
        
        log(f"   Запрос 3 (seed={seed3})...")
        r3 = call_model(model_key, task["prompt"], seed=seed3)
        results.append(r3)
    
    # Анализ результатов
    print("\n   📊 РЕЗУЛЬТАТЫ:")
    
    hashes = []
    for i, r in enumerate(results, 1):
        if r["success"]:
            h = compute_hash(r["content"])
            hashes.append(h)
            print(f"   Ответ {i}: {h[:16]}... | Токены: {r['usage'].get('total_tokens', 'N/A')} | Время: {r['elapsed']:.2f}с")
            print(f"            Модель: {r['model_used']}")
        else:
            hashes.append(None)
            print(f"   Ответ {i}: ОШИБКА - {r['error'][:50]}")
    
    # Проверка детерминизма
    print("\n   🔍 ПРОВЕРКА ДЕТЕРМИНИЗМА:")
    if hashes[0] and hashes[1]:
        if hashes[0] == hashes[1]:
            print(f"   ✅ Ответы 1 и 2 ИДЕНТИЧНЫ (хеши совпадают)")
        else:
            print(f"   ❌ Ответы 1 и 2 РАЗЛИЧАЮТСЯ (хеши НЕ совпадают)")
            print(f"      Hash1: {hashes[0]}")
            print(f"      Hash2: {hashes[1]}")
    
    if hashes[0] and hashes[2]:
        if hashes[0] != hashes[2]:
            print(f"   ✅ Ответ 3 ОТЛИЧАЕТСЯ от 1 и 2 (как и ожидалось)")
        else:
            print(f"   ⚠️  Ответ 3 СОВПАЛ с 1 (возможно случайность)")
    
    return results


def test_mcp_tools(model_key):
    """Тестирует поддержку MCP tool calling"""
    model = MODELS[model_key]
    log(f"🔧 Тест MCP Tools: {model['name']}")
    
    # System prompt нужен для надёжного вызова tools (особенно для Gemini)
    messages = [
        {"role": "system", "content": "Ты помощник для выполнения кода 1С. Используй инструмент execute_1c_code когда просят выполнить код."},
        {"role": "user", "content": "Выполни код 1С: СложитьЧисла(5, 3)"}
    ]
    
    request_body = {
        "model": model["id"],
        "messages": messages,
        "tools": [MCP_TOOL],
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=request_body,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        result = {"success": False, "elapsed": elapsed}
        
        if response.status_code == 200:
            data = response.json()
            choices = data.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                result = {
                    "success": True,
                    "content": message.get('content', ''),
                    "tool_calls": message.get('tool_calls', []),
                    "usage": data.get('usage', {}),
                    "elapsed": elapsed,
                    "model_used": data.get('model', model["id"])
                }
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text[:100]}"
            
    except Exception as e:
        result = {"success": False, "error": str(e), "elapsed": time.time() - start_time}
    
    print(f"\n   📊 РЕЗУЛЬТАТ MCP ТЕСТА:")
    if result["success"]:
        print(f"   Время: {result['elapsed']:.2f}с | Токены: {result['usage'].get('total_tokens', 'N/A')}")
        
        if result["tool_calls"]:
            print(f"   ✅ Модель вызвала tool!")
            for tc in result["tool_calls"]:
                func = tc.get('function', {})
                print(f"      Tool: {func.get('name')}")
                print(f"      Args: {func.get('arguments')}")
        else:
            print(f"   ⚠️  Модель НЕ вызвала tool")
            print(f"      Текстовый ответ: {result['content'][:200]}...")
    else:
        print(f"   ❌ ОШИБКА: {result.get('error', 'Unknown')}")
    
    return result


def main():
    print("=" * 80)
    print("🚀 ТЕСТИРОВАНИЕ ДЕТЕРМИНИЗМА И MCP ДЛЯ ИИ-МОДЕЛЕЙ")
    print("   Целевые модели: Claude Opus 4.5, GPT-5.2-Codex, Gemini 3 Flash")
    print("=" * 80)
    
    # 1. Начальный баланс
    print("\n📊 НАЧАЛЬНЫЙ БАЛАНС:")
    balance_start = get_balance()
    if balance_start:
        print(f"   Лимит: ${balance_start['limit']}")
        print(f"   Использовано: ${balance_start['usage']}")
        print(f"   Доступно: ${balance_start['available']}")
    
    # 2. Информация о моделях
    print("\n" + "=" * 80)
    print("📋 ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ О МОДЕЛЯХ:")
    print("=" * 80)
    
    for key, model in MODELS.items():
        info = get_model_info(model["id"])
        if info:
            print(f"\n🤖 {model['name']}:")
            print(f"   ID: {info['full_id']}")
            print(f"   Контекст: {info['context_length']:,} токенов")
            print(f"   Параметры: {', '.join(info['supported_parameters'][:10])}...")
            pricing = info['pricing']
            if pricing:
                inp = float(pricing.get('prompt', 0)) * 1_000_000
                out = float(pricing.get('completion', 0)) * 1_000_000
                print(f"   Цена: ${inp:.2f}/${out:.2f} за 1M токенов")
        else:
            print(f"\n⚠️  {model['name']}: информация недоступна")
    
    # 3. Тесты детерминизма
    print("\n" + "=" * 80)
    print("🧪 ТЕСТЫ ДЕТЕРМИНИЗМА:")
    print("=" * 80)
    
    all_results = {}
    
    for model_key in MODELS:
        print(f"\n{'─' * 60}")
        for task in TASKS_1C:
            results = test_determinism(model_key, task)
            all_results[f"{model_key}_{task['id']}"] = results
            print()
    
    # 4. Тесты MCP
    print("\n" + "=" * 80)
    print("🔧 ТЕСТЫ MCP TOOL CALLING:")
    print("=" * 80)
    
    mcp_results = {}
    for model_key in MODELS:
        print(f"\n{'─' * 60}")
        mcp_results[model_key] = test_mcp_tools(model_key)
    
    # 5. Итоговый баланс
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЙ БАЛАНС:")
    balance_end = get_balance()
    if balance_end and balance_start:
        spent = balance_end['usage'] - balance_start['usage']
        print(f"   Использовано: ${balance_end['usage']}")
        print(f"   Доступно: ${balance_end['available']}")
        print(f"   💰 Потрачено за сессию: ${spent}")
    
    # 6. Сводка
    print("\n" + "=" * 80)
    print("📝 СВОДКА ТЕСТИРОВАНИЯ:")
    print("=" * 80)
    
    print("\n🔹 Детерминизм:")
    for model_key, model in MODELS.items():
        param = model["determinism_param"]
        print(f"   {model['name']}: использует {param}")
    
    print("\n🔹 MCP Tool Calling:")
    for model_key, result in mcp_results.items():
        model = MODELS[model_key]
        if result["success"]:
            has_tools = "✅ ДА" if result.get("tool_calls") else "⚠️ НЕТ"
            print(f"   {model['name']}: {has_tools}")
        else:
            print(f"   {model['name']}: ❌ ОШИБКА")
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


if __name__ == "__main__":
    main()
