# %% [markdown]
# # Лабораторная работа 5: LangChain — создание AI-приложений на основе LLM
#
# ## Введение
#
# LangChain — экосистема для создания приложений на основе LLM.
# Вместо того чтобы писать всё с нуля, вы получаете набор готовых компонентов:
#
# * **Модели** — интеграция с Mistral, OpenAI, Anthropic и другими провайдерами
# * **Промпты** — шаблоны и техники эффективного общения с LLM
# * **Цепочки** — последовательность операций для решения сложных задач
# * **Агенты** — автономные системы, способные выбирать нужные инструменты
# * **Память** — сохранение контекста и истории взаимодействия
# * **Инструменты** — расширение возможностей LLM внешними функциями
#
# ## Чему вы научитесь
#
# 1. Архитектура LangChain — ключевые компоненты и принципы
# 2. Унифицированный интерфейс моделей — как работать с разными LLM через единый API
# 3. Инструменты (Tools) — как дать моделям новые способности
# 4. Intelligent Agents — как создавать умных агентов
# 5. Стратегии интеграции — разные подходы к работе с инструментами
# 6. Обработка и анализ данных — как сохранять и анализировать результаты
#
# ## Установка
#
# ```bash
# uv sync
# ```

# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
import pandas as pd
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# %% [markdown]
# ## 1. Что такое LangChain
#
# LangChain — это фреймворк-конструктор для AI-приложений.
# В следующем примере создадим простую цепочку:
# промпт -> модель -> парсер

# %%
def what_is_langchain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - эксперт по разработке на LangChain с глубоким пониманием архитектур LLM.

        Объясните в 3-4 предложениях, что такое LangChain и для чего он используется.
        Укажите основные компоненты и преимущества использования.
        """),
        ("human", "{input}")
    ])

    model = ChatMistralAI(model="mistral-small-latest", temperature=0)

    chain = prompt | model | StrOutputParser()

    response = chain.invoke({"input": "Что такое LangChain и для чего он используется?"})
    print(response)
    return response

print("\n=== 1. Что такое LangChain ===")
langchain_info = what_is_langchain()

# %% [markdown]
# ## 2. Унифицированный интерфейс для разных моделей
#
# Написав код один раз, можно использовать его с разными языковыми моделями
# без изменений. LangChain скрывает различия в API разных провайдеров
# и стандартизирует формат запросов и ответов.

# %%
def demonstrate_model_interface():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - эксперт по базам данных и технологиям AI.
        Объясните концепцию векторных баз данных кратко и по существу (2-3 предложения максимум)."""),
        ("human", "{question}")
    ])

    models = {
        "Mistral Small (temp=0)": ChatMistralAI(model="mistral-small-latest", temperature=0),
        "Mistral Small (temp=0.7)": ChatMistralAI(model="mistral-small-latest", temperature=0.7),
    }

    question = "Объясните концепцию векторных баз данных в 2-3 предложениях"

    results = {}
    for name, model in models.items():
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({"question": question})
        results[name] = response
        print(f"\n--- {name} ---")
        print(response)

    return results

print("\n=== 2. Единый интерфейс для разных моделей ===")
model_comparison = demonstrate_model_interface()

# %% [markdown]
# ## 3. Tools в LLM: расширяем возможности моделей
#
# Языковые модели ограничены генерацией текста. С помощью инструментов
# мы даём моделям возможность выполнять действия: искать информацию,
# вычислять формулы, вызывать API.

# %%
def demonstrate_tools():
    @tool
    def calculator(expression: str) -> float:
        """Вычисляет математическое выражение.

        Args:
            expression: Математическое выражение для вычисления (например, '2 + 2' или '3 * 5')

        Returns:
            Результат вычисления
        """
        try:
            result = eval(expression)
            print(f"[DEBUG] Калькулятор вычислил: {expression} = {result}")
            return result
        except Exception as e:
            error_msg = f"Ошибка при вычислении: {e}"
            print(f"[DEBUG] {error_msg}")
            return error_msg

    tools = [calculator]

    # 1. Прямое использование инструмента
    print("\n1. Прямое использование инструмента калькулятора:")
    calculation_result = calculator.invoke("12 * 34 + 5")
    print(f"Результат вычисления: {calculation_result}")

    # 2. Использование инструмента через модель
    print("\n2. Использование инструмента калькулятора через модель:")

    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    model_with_tools = model.bind_tools(tools)

    system_message = """Вы - ассистент-калькулятор, который ВСЕГДА использует предоставленный инструмент калькулятора.

    ИНСТРУКЦИИ:
    1. ВСЕГДА используйте инструмент калькулятора для выполнения математических вычислений
    2. НИКОГДА не пытайтесь вычислить ответ самостоятельно
    3. Определите математическое выражение в запросе пользователя
    4. Вызовите инструмент калькулятора с этим выражением
    5. Верните результат от калькулятора"""

    query = "25 * 16 - 38"

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"Решите это математическое выражение: {query}")
    ]

    response = model_with_tools.invoke(messages)

    print("\nОтвет модели:")
    tool_calls = getattr(response, "tool_calls", [])

    if not response.content and tool_calls:
        print("Модель решила использовать инструмент вместо прямого ответа.")
    else:
        print(response.content)

    if tool_calls:
        print("\nИнструменты, вызванные моделью:")
        for call in tool_calls:
            print(f"- Инструмент: {call['name']}")
            print(f"  Аргументы: {call['args']}")

            if call['name'] == 'calculator':
                tool_result = calculator.invoke(call['args']['expression'])
                print(f"  Результат выполнения: {tool_result}")

                if not response.content:
                    print(f"\nСформированный ответ на основе вызова инструмента:")
                    print(f"Для решения выражения {query} был использован калькулятор.")
                    print(f"Результат: {tool_result}")
    else:
        print("\nМодель не вызвала инструменты.")

    # 3. Полный цикл: вызов инструмента и возврат результата в модель
    print("\n3. Полный цикл использования инструментов:")

    new_query = "Посчитай площадь круга с радиусом 7 см"

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=new_query)
    ]

    first_response = model_with_tools.invoke(messages)
    messages.append(first_response)

    tool_calls = getattr(first_response, "tool_calls", [])

    if tool_calls:
        from langchain_core.messages import ToolMessage

        for call in tool_calls:
            if call['name'] == 'calculator':
                expression = call['args'].get('expression')
                if expression:
                    tool_result = calculator.invoke(expression)
                    print(f"Выполнен инструмент {call['name']} с аргументом {expression}, результат: {tool_result}")

                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=call['id']
                        )
                    )

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

print("\n=== 3. Tools в LLM ===")
tool_demonstration = demonstrate_tools()

# %% [markdown]
# ## 4. Создание AI-агентов
#
# Агенты — самый продвинутый способ использования LangChain.
# Агент самостоятельно принимает решения о том, какие инструменты
# использовать для решения поставленной задачи.

# %%
def demonstrate_tool_interface():
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

    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
    tools_list = [get_weather, get_population, TavilySearch(max_results=2)]

    agent = create_agent(
        llm,
        tools_list,
        system_prompt="Вы - информационный ассистент с доступом к различным инструментам для сбора данных. "
                      "Используйте доступные инструменты для получения точной информации и ответа на вопросы пользователя."
    )

    print("\nЗапрос агенту: 'Какая погода в Москве и сколько там жителей?'")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Какая погода в Москве и сколько там жителей?"}]
    })

    # Извлекаем последний ответ из messages
    last_message = response["messages"][-1]
    output = last_message.content if hasattr(last_message, "content") else str(last_message)

    return {
        "input": "Какая погода в Москве и сколько там жителей?",
        "output": output,
        "tools_used": [t.name for t in tools_list]
    }

print("\n=== 4. Интерфейс tool в LangChain ===")
agent_demonstration = demonstrate_tool_interface()

# %% [markdown]
# ## 5. Разделение ответственности: контроль над инструментами
#
# Два подхода к работе с LLM:
#
# 1. **Автономный подход** — модель сама решает, когда вызывать инструменты
# 2. **Контролируемый подход** — вы явно вызываете инструменты и формируете контекст
#
# Контролируемый подход полезен при работе с конфиденциальными данными,
# при необходимости точного контроля над источниками информации
# и для создания приложений с предсказуемым поведением.

# %%
def demonstrate_two_tool_approaches():
    search = TavilySearch(max_results=3)
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

    query = "Какие ключевые компоненты и возможности предлагает фреймворк LangChain?"

    print("\nПодход: Вызов инструмента с последующим обогащением контекста")
    print(f"Шаг 1: Выполняем поисковый запрос: '{query}'")

    search_results = search.invoke(query)

    # TavilySearch возвращает dict с ключом 'results'
    results_list = search_results.get("results", []) if isinstance(search_results, dict) else search_results

    print("\nШаг 2: Получаем результаты поиска (первые 2 результата):")
    for i, result in enumerate(results_list[:2]):
        print(f"\nИсточник {i+1}: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Фрагмент: {result['content'][:150]}...")

    print("\nШаг 3: Передаём результаты поиска модели через контекст")
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - эксперт по разработке на LangChain.

        ИНСТРУКЦИИ:
        1. Используйте ТОЛЬКО предоставленный контекст для ответа на вопрос.
        2. Если в контексте недостаточно информации, так и скажите.
        3. Структурируйте ответ в виде маркированного списка.
        4. Объясняйте технические концепции простым языком.
        """),
        ("human", "Контекст из поиска:\n\n{context}\n\nВопрос: {query}\n\nДайте структурированный ответ на основе этого контекста.")
    ])

    chain = context_prompt | llm | StrOutputParser()

    print("\nШаг 4: Модель формирует ответ на основе контекста")
    response = chain.invoke({
        "context": str(search_results),
        "query": query
    })

    print("\nРезультат:")
    print(response)

    return {
        "query": query,
        "approach_response": response
    }

print("\n=== 5. Разделение ответственности ===")
tool_approaches = demonstrate_two_tool_approaches()

# %% [markdown]
# ## 6. Анализ и хранение данных
#
# Создание AI-решений — итеративный процесс.
# Сохраняем результаты в CSV через Pydantic + Pandas,
# затем анализируем с помощью LLM.

# %%
class ExperimentResult(BaseModel):
    section: str = Field(description="Раздел эксперимента")
    query: str = Field(description="Запрос к системе")
    response: str = Field(description="Ответ модели или инструмента")
    model: str = Field(description="Название использованной модели")
    approach: str = Field(description="Метод или техника")
    prompt_technique: Optional[str] = Field(description="Техника промптирования", default=None)


def save_results_to_csv():
    results = [
        ExperimentResult(
            section="Что такое LangChain",
            query="Что такое LangChain и для чего он используется?",
            response=langchain_info,
            model="mistral-small-latest",
            approach="basic",
            prompt_technique="Базовый промпт"
        ),
        ExperimentResult(
            section="Единый интерфейс",
            query="Объясните концепцию векторных баз данных",
            response=model_comparison["Mistral Small (temp=0)"],
            model="Mistral Small (temp=0)",
            approach="model_comparison",
            prompt_technique="Стандартный промпт"
        ),
        ExperimentResult(
            section="Единый интерфейс",
            query="Объясните концепцию векторных баз данных",
            response=model_comparison["Mistral Small (temp=0.7)"],
            model="Mistral Small (temp=0.7)",
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
            model="mistral-small-latest",
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
            model="mistral-small-latest",
            approach="separate_tool_call",
            prompt_technique="Отдельный вызов инструмента"
        )
    ]

    df = pd.DataFrame([result.model_dump() for result in results])

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - аналитик данных, специализирующийся на технологиях LLM.

        Проанализируйте результаты экспериментов с LangChain:
        {experiment_data}

        Определите 3 главных вывода о преимуществах использования LangChain.
        Представьте выводы в виде четких, кратких пунктов."""),
        ("human", "Проведите анализ результатов экспериментов.")
    ])

    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    chain = analysis_prompt | model | StrOutputParser()

    analysis = chain.invoke({"experiment_data": df.to_string()})
    print("\nАнализ результатов экспериментов:")
    print(analysis)

    csv_filename = "langchain_results.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')

    print(f"\nРезультаты сохранены в {csv_filename}")
    print("\nПример содержимого DataFrame:")
    print(df[["section", "query", "model", "prompt_technique"]].head())

    return df

print("\n=== 6. Сохранение результатов в CSV ===")
results_df = save_results_to_csv()

# %% [markdown]
# ## Выводы
#
# После выполнения лабораторной работы ответьте на вопросы:
#
# 1. Какие преимущества даёт единый интерфейс LangChain для разных моделей?
# 2. В каких случаях лучше использовать автономный подход (агенты), а в каких — контролируемый?
# 3. Как инструменты расширяют возможности языковых моделей?
# 4. Какие практические применения вы видите для агентных систем?
#
# ---
#
# ## Ваши наблюдения
#
# *Запишите здесь свои выводы*

# %%
# Место для дополнительных экспериментов
