#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Пример использования LangChain с различными языковыми моделями,
инструментами и сохранением результатов в CSV с помощью pandas.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any

# Загрузка переменных окружения из файла .env
load_dotenv()

# Импорт основных компонентов LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# Импорт модуля для работы с различными моделями
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

# Импорт инструментов, включая Tavily для поиска в интернете
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import Tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor


# Функция для сохранения результатов в CSV
def save_results_to_csv(results: List[Dict[str, Any]], filename: str = "results.csv"):
    """
    Сохраняет результаты в CSV файл с помощью pandas
    
    Args:
        results: Список словарей с результатами
        filename: Имя файла для сохранения
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Результаты сохранены в {filename}")
    return df


# Функция для демонстрации разницы между моделями
def compare_models():
    """Сравнение нескольких моделей с использованием единого интерфейса LangChain"""
    
    # Создаем единый шаблон запроса для всех моделей
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы - полезный ассистент. Ответьте коротко на вопрос пользователя."),
        ("human", "{question}")
    ])
    
    # Инициализация разных моделей через единый интерфейс
    models = {
        "OpenAI GPT-3.5-Turbo": ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        "OpenAI GPT-4": ChatOpenAI(model_name="gpt-4", temperature=0),
    }
    
    # Вопрос для всех моделей
    question = "Что такое LangChain и для чего он используется?"
    
    # Сравнение ответов разных моделей
    results = []
    
    for model_name, model in models.items():
        # Создаем цепочку для получения ответа
        chain = prompt | model | StrOutputParser()
        
        # Получаем ответ
        response = chain.invoke({"question": question})
        
        # Добавляем результат в список
        results.append({"model": model_name, "question": question, "response": response})
        
        # Выводим результат
        print(f"\n--- {model_name} ---")
        print(response)
    
    # Сохраняем результаты
    save_results_to_csv(results, "model_comparison.csv")
    
    return results


# Функция для демонстрации использования инструментов (tools)
def use_tools():
    """Демонстрация использования инструментов в LangChain"""
    
    # Инициализация инструмента Tavily для поиска в интернете
    search_tool = TavilySearchResults(
        max_results=3,  # Количество результатов поиска
        include_raw_content=True,  # Включить сырое содержимое
        include_images=False,  # Не включать изображения
    )
    
    # Создаем список инструментов
    tools = [search_tool]
    
    # Инициализация модели для агента
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Создаем системный промпт
    system_prompt = """
    Вы - полезный ассистент. Используйте инструменты, доступные вам, 
    чтобы ответить на вопрос пользователя. Если нужно найти актуальную 
    информацию, используйте инструмент поиска.
    
    Ответ должен быть на русском языке.
    """
    
    # Создаем агента с инструментами
    agent = create_openai_tools_agent(llm, tools, system_prompt)
    
    # Создаем исполнителя агента
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Список запросов для тестирования
    queries = [
        "Какие основные компоненты есть в LangChain?",
        "Что такое Retrieval Augmented Generation (RAG) и как это связано с LangChain?",
        "Что нового произошло в сфере AI за последний месяц?"
    ]
    
    # Выполняем запросы и сохраняем результаты
    results = []
    
    for query in queries:
        print(f"\n\nЗапрос: {query}")
        
        # Выполняем запрос через агента
        response = agent_executor.invoke({"input": query})
        
        # Добавляем результат в список
        results.append({
            "query": query, 
            "response": response["output"],
            "tool_used": "Tavily Search"
        })
    
    # Сохраняем результаты
    save_results_to_csv(results, "tool_results.csv")
    
    return results


# Функция для демонстрации двух способов использования инструментов
def tools_usage_comparison():
    """
    Сравнение двух подходов к использованию инструментов:
    1. Отдельный вызов инструмента и добавление результатов в контекст
    2. Интеграция инструмента напрямую в цепочку вызова модели (bind tools)
    """
    
    # Инициализация инструмента Tavily
    search_tool = TavilySearchResults(max_results=2)
    
    # Инициализация модели
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Запрос для тестирования
    query = "Назови три последние модели крупных языковых моделей и их особенности"
    
    print("=== Подход 1: Отдельный вызов инструмента ===")
    
    # 1. Отдельный вызов инструмента и затем модели
    search_results = search_tool.invoke(query)
    
    # Формируем запрос с контекстом из результатов поиска
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты - полезный ассистент. Используй информацию из контекста для ответа на вопрос."),
        ("human", "Контекст: {context}\n\nВопрос: {question}")
    ])
    
    # Создаем цепочку
    separate_chain = context_prompt | llm | StrOutputParser()
    
    # Получаем ответ
    separate_response = separate_chain.invoke({
        "context": str(search_results), 
        "question": query
    })
    
    print("\nРезультат подхода 1 (отдельный вызов):")
    print(separate_response)
    
    print("\n=== Подход 2: Интеграция инструмента (bind tools) ===")
    
    # 2. Использование инструмента через bind_tools
    llm_with_tools = llm.bind_tools([search_tool])
    
    # Создаем промпт
    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты - полезный ассистент. Используй доступные инструменты, чтобы найти актуальную информацию."),
        ("human", "{question}")
    ])
    
    # Создаем цепочку
    direct_chain = direct_prompt | llm_with_tools | StrOutputParser()
    
    # Получаем ответ
    direct_response = direct_chain.invoke({"question": query})
    
    print("\nРезультат подхода 2 (bind tools):")
    print(direct_response)
    
    # Сохраняем результаты
    results = [
        {"method": "Отдельный вызов инструмента", "query": query, "response": separate_response},
        {"method": "Интеграция инструмента (bind tools)", "query": query, "response": direct_response}
    ]
    
    save_results_to_csv(results, "tool_methods_comparison.csv")
    
    return results


# Интерактивный режим
def interactive_mode():
    """Интерактивный режим для тестирования LangChain"""
    
    # Инициализация инструмента Tavily
    search_tool = TavilySearchResults(max_results=3)
    
    # Создаем инструменты
    tools = [search_tool]
    
    # Инициализация модели
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Создаем системный промпт
    system_prompt = """
    Вы - полезный ассистент. Используйте инструменты, доступные вам, 
    чтобы ответить на вопрос пользователя. Если нужно найти актуальную 
    информацию, используйте инструмент поиска.
    
    Ответ должен быть на русском языке.
    """
    
    # Создаем агента с инструментами
    agent = create_openai_tools_agent(llm, tools, system_prompt)
    
    # Создаем исполнителя агента
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Список результатов для сохранения
    results = []
    
    print("\n=== Интерактивный режим LangChain с Tavily Search ===")
    print("Введите ваш запрос или 'выход' для завершения.")
    
    while True:
        query = input("\nВаш запрос: ")
        
        if query.lower() in ["выход", "exit", "quit"]:
            break
        
        # Выполняем запрос через агента
        response = agent_executor.invoke({"input": query})
        
        # Добавляем результат в список
        results.append({
            "query": query, 
            "response": response["output"],
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Если были запросы, сохраняем результаты
    if results:
        save_results_to_csv(results, "interactive_results.csv")
    
    return results


# Основная функция для запуска демонстраций
def main():
    """Основная функция для демонстрации возможностей LangChain"""
    
    print("\n===== LangChain: демонстрация возможностей =====\n")
    
    while True:
        print("\nВыберите демонстрацию:")
        print("1. Сравнение разных моделей через единый интерфейс")
        print("2. Использование инструментов (Tavily Search)")
        print("3. Сравнение подходов к использованию инструментов")
        print("4. Интерактивный режим")
        print("0. Выход")
        
        choice = input("\nВаш выбор: ")
        
        if choice == "1":
            compare_models()
        elif choice == "2":
            use_tools()
        elif choice == "3":
            tools_usage_comparison()
        elif choice == "4":
            interactive_mode()
        elif choice == "0":
            print("\nЗавершение работы.")
            break
        else:
            print("\nНеверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
