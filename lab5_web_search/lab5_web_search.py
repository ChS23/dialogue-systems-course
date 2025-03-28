# %% [markdown]
# # Лабораторная работа: LangChain - основные возможности и практическое применение
# 
# ## Введение
# 
# Добро пожаловать в лабораторную работу по фреймворку LangChain! В этой лаборатории мы погрузимся в мир современных инструментов для работы с большими языковыми моделями (LLM) и научимся создавать эффективные приложения на их основе.
# 
# LangChain - это один из наиболее мощных и популярных фреймворков для работы с LLM, который позволяет существенно расширить возможности моделей и интегрировать их с внешними источниками данных и инструментами.
# 
# ## Что вы изучите в этой лабораторной работе:
# 
# 1. **Основы LangChain** - архитектура, компоненты и ключевые концепции
# 2. **Универсальный интерфейс для моделей** - как использовать один код с разными LLM
# 3. **Инструменты и их применение** - расширение возможностей моделей с помощью внешних функций
# 4. **Агенты и интерфейсы инструментов** - создание интеллектуальных агентов, способных самостоятельно решать задачи
# 5. **Разные подходы к работе с инструментами** - сравнение методов интеграции
# 6. **Обработка и анализ результатов** - сохранение данных и их анализ
# 


# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Загрузка переменных окружения
load_dotenv(".env")

# Импорт компонентов LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# %% [markdown]
# ## 1. Что такое LangChain
# 
# LangChain - это фреймворк для разработки приложений, основанных на больших языковых моделях (LLM). 
# Он предоставляет набор компонентов и инструментов для создания сложных цепочек обработки, 
# которые сочетают языковые модели с другими источниками данных и вычислениями.

# %%
# Демонстрация основной функциональности LangChain
def what_is_langchain():
    # Создаем простой промпт
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - эксперт по разработке на LangChain с глубоким пониманием архитектур LLM.
        
        Объясните в 3-4 предложениях, что такое LangChain и для чего он используется.
        Укажите основные компоненты и преимущества использования.
        """
        ),
        ("human", "{input}")
    ])
    
    # Инициализируем модель
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Создаем цепочку: промпт -> модель -> парсер
    # Это основной паттерн работы с LangChain
    chain = prompt | model | StrOutputParser()
    
    # Выполняем запрос и получаем ответ
    response = chain.invoke({"input": "Что такое LangChain и для чего он используется?"})
    print(response)
    return response

# Запускаем функцию
print("\n=== 1. Что такое LangChain ===")
langchain_info = what_is_langchain()

# %% [markdown]
# ## 2. Единый интерфейс для разных моделей
# 
# Одно из главных преимуществ LangChain - возможность использовать один и тот же код 
# с разными языковыми моделями. Это обеспечивает гибкость при разработке и позволяет
# легко переключаться между моделями без изменения основного кода приложения.

# %%
# Демонстрация единого интерфейса для разных моделей
def demonstrate_model_interface():
    # Создаем простой промпт, который будет использоваться с разными моделями
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - эксперт по базам данных и технологиям AI. 
        Объясните концепцию векторных баз данных кратко и по существу (2-3 предложения максимум)."""),
        ("human", "{question}")
    ])
    
    # Инициализируем разные модели через единый интерфейс
    # В реальном приложении можно использовать и другие модели (Claude, Llama, и т.д.)
    models = {
        "GPT-4o-mini (temp=0)": ChatOpenAI(model="gpt-4o-mini", temperature=0),
        "GPT-4o-mini (temp=0.7)": ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    }
    
    question = "Объясните концепцию векторных баз данных в 2-3 предложениях"
    
    # Один и тот же код работает с разными моделями
    results = {}
    for name, model in models.items():
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({"question": question})
        results[name] = response
        print(f"\n--- {name} ---")
        print(response)
    
    return results

# Запускаем функцию
print("\n=== 2. Единый интерфейс для разных моделей ===")
model_comparison = demonstrate_model_interface()

# %% [markdown]
# ## 3. Tools в LLM
# 
# LangChain позволяет расширять возможности языковых моделей с помощью инструментов (tools).
# Инструменты - это функции, которые модель может использовать для выполнения определенных 
# действий, таких как вычисления, поиск информации или доступ к внешним API.
#
# В этом разделе мы рассмотрим:
# - Как создать собственные инструменты с помощью декоратора `@tool`
# - Как использовать готовые инструменты из библиотеки LangChain
# - Как модель вызывает инструменты и как обрабатывать результаты этих вызовов
#
# **Важно:** При запуске кода вы увидите следующие результаты:
# ```
# Результат вычисления: 413  # Результат операции 12 * 34 + 5
# Результат поиска: информация из интернета о LangChain
# ```
# 
# Эти результаты показывают, что инструменты работают корректно и возвращают ожидаемые данные.
# Инструмент поиска TavilySearchResults выполняет реальный поиск в интернете и может возвращать
# разные результаты в зависимости от доступности данных и обновления источников.

# %%
# Демонстрация работы с инструментами в LangChain
def demonstrate_tools():
    # Создаем инструмент калькулятора
    @tool
    def calculator(expression: str) -> float:
        """Вычисляет математическое выражение.
        
        Args:
            expression: Математическое выражение для вычисления (например, '2 + 2' или '3 * 5')
        
        Returns:
            Результат вычисления
        """
        try:
            # Вычисление выражения
            result = eval(expression)
            print(f"[DEBUG] Калькулятор вычислил: {expression} = {result}")
            return result
        except Exception as e:
            error_msg = f"Ошибка при вычислении: {e}"
            print(f"[DEBUG] {error_msg}")
            return error_msg
    
    # Формируем список инструментов
    tools = [calculator]
    
    # 1. Базовая демонстрация прямого использования инструмента калькулятора
    print("\n1. Прямое использование инструмента калькулятора:")
    calculation_result = calculator.invoke("12 * 34 + 5")
    print(f"Результат вычисления: {calculation_result}")
 
    # 2. Демонстрация использования инструмента калькулятора через модель
    print("\n2. Использование инструмента калькулятора через модель:")
    
    # Создаем модель с температурой 0 для детерминированных ответов
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Привязываем инструменты к модели с помощью .bind_tools()
    # Это современный метод в LangChain для передачи схем инструментов модели
    model_with_tools = model.bind_tools(tools)
    
    # Создаем системное сообщение с инструкциями для модели
    system_message = """Вы - ассистент-калькулятор, который ВСЕГДА использует предоставленный инструмент калькулятора.
    
    ИНСТРУКЦИИ:
    1. ВСЕГДА используйте инструмент калькулятора для выполнения математических вычислений
    2. НИКОГДА не пытайтесь вычислить ответ самостоятельно
    3. Определите математическое выражение в запросе пользователя
    4. Вызовите инструмент калькулятора с этим выражением
    5. Верните результат от калькулятора
    
    ФОРМАТ ОТВЕТА:
    Для решения я использовал выражение: [выражение]
    Результат: [результат]
    
    ПРИМЕР:
    Пользователь: Посчитай 5 * 10
    Вы: Для решения я использовал выражение: 5 * 10
    Результат: 50
    
    Теперь решите математическую задачу пользователя, используя инструмент калькулятора."""
    
    # Создаем явный запрос на вычисление
    query = "25 * 16 - 38"
    
    # Создаем сообщения для отправки модели
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"Решите это математическое выражение: {query}")
    ]
    
    # Вызываем модель с инструментами и получаем ответ
    response = model_with_tools.invoke(messages)
    
    # Проверяем наличие вызовов инструментов (tool_calls) в ответе
    print("\nОтвет модели:")
    tool_calls = getattr(response, "tool_calls", [])
    
    # Когда модель вызывает инструменты, поле content может быть пустым - это нормально
    # В этом случае мы выводим информацию о вызове инструмента
    if not response.content and tool_calls:
        print("Модель решила использовать инструмент вместо прямого ответа.")
    else:
        print(response.content)
    
    if tool_calls:
        print("\nИнструменты, вызванные моделью:")
        for call in tool_calls:
            print(f"- Инструмент: {call['name']}")
            print(f"  Аргументы: {call['args']}")
            
            # Демонстрация выполнения вызванного инструмента
            if call['name'] == 'calculator':
                tool_result = calculator.invoke(call['args']['expression'])
                print(f"  Результат выполнения: {tool_result}")
                
                # Если content пуст, мы можем сформировать ответ на основе вызова инструмента
                if not response.content:
                    print(f"\nСформированный ответ на основе вызова инструмента:")
                    print(f"Для решения выражения {query} был использован калькулятор.")
                    print(f"Результат: {tool_result}")
    else:
        print("\nМодель не вызвала инструменты.")
    
    # 3. Демонстрация полного цикла (вызов инструмента и передача результата обратно в модель)
    print("\n3. Полный цикл использования инструментов (вызов и возврат результата в модель):")
    
    # Создаем новый запрос
    new_query = "Посчитай площадь круга с радиусом 7 см"
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=new_query)
    ]
    
    # Вызываем модель для получения tool_calls
    first_response = model_with_tools.invoke(messages)
    
    # Теперь нам нужно добавить это сообщение в историю
    messages.append(first_response)
    
    tool_calls = getattr(first_response, "tool_calls", [])
    
    # Выполняем инструменты, вызванные моделью
    if tool_calls:
        for call in tool_calls:
            if call['name'] == 'calculator':
                expression = call['args'].get('expression')
                if expression:
                    tool_result = calculator.invoke(expression)
                    print(f"Выполнен инструмент {call['name']} с аргументом {expression}, результат: {tool_result}")
                    
                    # Добавляем результат инструмента в историю как ToolMessage
                    # Это правильный способ предоставить результаты инструмента обратно модели
                    from langchain_core.messages import ToolMessage
                    
                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=call['id']
                        )
                    )
        
        # Получаем финальный ответ от модели после вызова инструментов
        final_response = model.invoke(messages)
        print("\nФинальный ответ модели после получения результатов от инструментов:")
        print(final_response.content)
    else:
        print("\nМодель не вызвала инструменты в этом запросе.")
    
    return {
        "calculation": calculation_result,
        "model_tool_usage": response.content,
        "tool_calls": tool_calls
    }

# Запускаем функцию
print("\n=== 3. Tools в LLM ===")
print("В этом разделе демонстрируется создание и использование инструментов в LangChain.")
print("Ниже вы увидите примеры использования инструментов напрямую и через модель.")
tool_demonstration = demonstrate_tools()

# %% [markdown]
# ## 4. Интерфейс tool в LangChain
# 
# LangChain предоставляет удобный интерфейс для создания и использования инструментов.
# Один из наиболее мощных способов использования инструментов - это создание агентов,
# которые могут самостоятельно выбирать и применять нужные инструменты для решения задач.

# %%
# Демонстрация интерфейса инструментов в LangChain
def demonstrate_tool_interface():
    # Создаем инструменты
    @tool
    def get_weather(location: str) -> str:
        """Получает текущую погоду в указанном месте."""
        weather_data = {
            "Москва": "Облачно, 18°C",
            "Санкт-Петербург": "Дождь, 15°C",
            "Нью-Йорк": "Солнечно, 25°C",
            "Лондон": "Туман, 14°C"
        }
        return weather_data.get(location, f"Информация о погоде для {location} недоступна")
    
    @tool
    def get_population(city: str) -> str:
        """Получает информацию о населении города."""
        population_data = {
            "Москва": "12 млн человек",
            "Санкт-Петербург": "5 млн человек",
            "Нью-Йорк": "8 млн человек",
            "Лондон": "9 млн человек"
        }
        return population_data.get(city, f"Информация о населении {city} недоступна")
    
    # Создаем агента с доступом к инструментам
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools_list = [get_weather, get_population, TavilySearchResults(max_results=2)]
    
    # Используем встроенный промпт вместо создания своего
    # Это самый простой способ обеспечить правильное форматирование
    agent = create_openai_tools_agent(
        llm,
        tools_list,
        ChatPromptTemplate.from_messages([
            ("system", "Вы - информационный ассистент с доступом к различным инструментам для сбора данных. "
                      "Вы должны использовать доступные инструменты для получения точной информации и ответа на вопросы пользователя. "
                      "Анализируйте вопросы, выбирайте подходящие инструменты и предоставляйте четкие ответы."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
    )
    
    # Создаем исполнителя агента
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print("\nЗапрос агенту: 'Какая погода в Москве и сколько там жителей?'")
    response = agent_executor.invoke({
        "input": "Какая погода в Москве и сколько там жителей?"
    })
    
    return {
        "input": "Какая погода в Москве и сколько там жителей?",
        "output": response["output"],
        "tools_used": [tool.name for tool in tools_list]
    }

# Запускаем функцию
print("\n=== 4. Интерфейс tool в LangChain ===")
agent_demonstration = demonstrate_tool_interface()

# %% [markdown]
# ## 5. Разделение ответственности: вызов инструмента и обогащение контекста
# 
# В отличие от предыдущих примеров, где модель сама решает когда вызывать инструменты,
# иногда нам нужно явно контролировать этот процесс. Для сложных задач часто предпочтительнее
# разделить ответственность:
# 1. Сначала мы сами вызываем инструмент для получения информации
# 2. Затем передаём эту информацию модели как часть контекста для формирования ответа
#
# Этот подход даёт нам больше контроля над процессом и позволяет комбинировать данные из разных источников.

# %%
# Демонстрация подхода с разделением ответственности
def demonstrate_two_tool_approaches():
    # Создаем инструмент для поиска
    search = TavilySearchResults(max_results=3)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Формулируем информативный запрос для поиска
    query = "Какие ключевые компоненты и возможности предлагает фреймворк LangChain?"
    
    print("\nПодход: Вызов инструмента с последующим обогащением контекста")
    print(f"Шаг 1: Выполняем поисковый запрос: '{query}'")
    
    # Вызываем инструмент напрямую
    search_results = search.invoke(query)
    
    # Выводим фрагменты результатов поиска
    print("\nШаг 2: Получаем результаты поиска (первые 2 результата):")
    for i, result in enumerate(search_results[:2]):
        print(f"\nИсточник {i+1}: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Фрагмент: {result['content'][:150]}...")
    
    # Создаем промпт, который явно инструктирует модель, как использовать предоставленный контекст
    print("\nШаг 3: Передаем результаты поиска модели через контекст")
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - эксперт по разработке на LangChain. 
        
        ИНСТРУКЦИИ:
        1. Используйте ТОЛЬКО предоставленный контекст для ответа на вопрос.
        2. Если в контексте недостаточно информации, так и скажите.
        3. Структурируйте ответ в виде маркированного списка основных компонентов и возможностей.
        4. Объясняйте технические концепции простым языком.
        """),
        ("human", "Контекст из поиска:\n\n{context}\n\nВопрос: {query}\n\nПожалуйста, дайте структурированный ответ на основе этого контекста.")
    ])
    
    # Формируем цепочку: промпт с контекстом -> модель -> парсер
    chain = context_prompt | llm | StrOutputParser()
    
    # Выполняем запрос с контекстом
    print("\nШаг 4: Модель формирует ответ на основе контекста")
    response = chain.invoke({
        "context": str(search_results), 
        "query": query
    })
    
    print("\nРезультат:")
    print(response)
    
    print("\nПреимущества этого подхода:")
    print("1. Полный контроль над процессом получения данных")
    print("2. Возможность предобработки и фильтрации результатов перед отправкой модели")
    print("3. Возможность комбинировать данные из разных источников")
    print("4. Более прозрачный и предсказуемый результат")
    
    return {
        "query": query,
        "approach_response": response
    }

# Запускаем функцию
print("\n=== 5. Разделение ответственности: вызов инструмента и обогащение контекста ===")
tool_approaches = demonstrate_two_tool_approaches()

# %% [markdown]
# ## 6. Сохранение результатов с pandas в CSV
# 
# LangChain отлично интегрируется с экосистемой Python для анализа данных.
# В этом разделе мы продемонстрируем, как сохранять результаты работы LLM 
# и инструментов в структурированном формате с помощью pandas и CSV.

# %%
# Определяем структуру данных через Pydantic
class ExperimentResult(BaseModel):
    section: str = Field(description="Раздел эксперимента")
    query: str = Field(description="Запрос к системе")
    response: str = Field(description="Ответ модели или инструмента")
    model: str = Field(description="Название использованной модели")
    approach: str = Field(description="Метод или техника, использованная в эксперименте")
    prompt_technique: Optional[str] = Field(description="Техника промптирования, если применялась", default=None)

# Сохранение результатов в CSV
def save_results_to_csv():
    # Собираем результаты всех экспериментов
    results = [
        ExperimentResult(
            section="Что такое LangChain",
            query="Что такое LangChain и для чего он используется?",
            response=langchain_info,
            model="gpt-4o-mini",
            approach="basic",
            prompt_technique="Базовый промпт"
        ),
        ExperimentResult(
            section="Единый интерфейс",
            query="Объясните концепцию векторных баз данных",
            response=model_comparison["GPT-4o-mini (temp=0)"],
            model="GPT-4o-mini (temp=0)",
            approach="model_comparison",
            prompt_technique="Стандартный промпт"
        ),
        ExperimentResult(
            section="Единый интерфейс",
            query="Объясните концепцию векторных баз данных",
            response=model_comparison["GPT-4o-mini (temp=0.7)"],
            model="GPT-4o-mini (temp=0.7)",
            approach="model_comparison",
            prompt_technique="Стандартный промпт"
        ),
        ExperimentResult(
            section="Tools в LLM",
            query="12 * 34 + 5",
            response=str(tool_demonstration["calculation"]),
            model="calculator",
            approach="direct_tool",
            prompt_technique="Прямой вызов инструмента"
        ),
        ExperimentResult(
            section="Tools в LLM (через модель)",
            query="25 * 16 - 38",
            response=tool_demonstration["model_tool_usage"],
            model="gpt-4o-mini",
            approach="model_with_tools",
            prompt_technique="Использование инструмента через модель"
        ),
        ExperimentResult(
            section="Интерфейс tool",
            query="Какая погода в Москве и сколько там жителей?",
            response=agent_demonstration["output"],
            model="agent",
            approach="agent_with_tools",
            prompt_technique="Агентный подход"
        ),
        ExperimentResult(
            section="Два подхода к использованию инструментов",
            query=tool_approaches["query"],
            response=tool_approaches["approach_response"],
            model="gpt-4o-mini",
            approach="separate_tool_call",
            prompt_technique="Отдельный вызов инструмента"
        )
    ]
    
    # Создаем DataFrame
    df = pd.DataFrame([result.model_dump() for result in results])
    
    # Анализ результатов с использованием LLM
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - аналитик данных, специализирующийся на технологиях LLM.
        
        Проанализируйте результаты экспериментов с LangChain:
        {experiment_data}
        
        Определите 3 главных вывода о преимуществах использования LangChain для работы с языковыми моделями.
        Представьте выводы в виде четких, кратких пунктов."""),
        ("human", "Проведите анализ результатов экспериментов.")
    ])
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = analysis_prompt | model | StrOutputParser()
    
    analysis = chain.invoke({"experiment_data": df.to_string()})
    print("\nАнализ результатов экспериментов:")
    print(analysis)
    
    # Сохраняем в CSV
    csv_filename = "langchain_results.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"Результаты сохранены в {csv_filename}")
    print("\nПример содержимого DataFrame:")
    print(df[["section", "query", "model", "prompt_technique"]].head())
    
    return df

# Запускаем функцию
print("\n=== 6. Сохранение результатов с pandas в CSV ===")
results_df = save_results_to_csv()

# %% [markdown]
# ## Заключение
# 
# В этой лабораторной работе мы познакомились с основными возможностями фреймворка LangChain:
# 
# 1. **Общее понимание LangChain** - фреймворк для создания приложений на основе LLM с гибкой архитектурой компонентов
# 2. **Единый интерфейс для разных моделей** - возможность использовать один код с разными языковыми моделями
# 3. **Инструменты (Tools) в LLM** - расширение возможностей языковых моделей с помощью функций и внешних API
# 4. **Интерфейс инструментов** - создание и использование агентов, способных выбирать и применять нужные инструменты
# 5. **Подходы к интеграции инструментов** - отдельные вызовы и прямая привязка инструментов к модели
# 6. **Работа с данными** - сохранение и анализ результатов с использованием pandas и CSV
# 
# Дополнительно, мы познакомились с различными техниками промптинга, которые повышают эффективность работы с языковыми моделями и делают результаты более предсказуемыми и качественными.
