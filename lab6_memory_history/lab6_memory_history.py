# %% [markdown]
# # Лабораторная работа 6: Память и управление состоянием в диалоговых системах
#
# ## Введение
#
# Диалоговая система без памяти не может поддерживать осмысленный разговор.
# Для создания полезного AI-ассистента необходимо:
# * Хранить историю взаимодействия с пользователем
# * Управлять состоянием диалога
# * Организовывать долгосрочную и краткосрочную память
#
# В этой лабораторной мы используем **LangChain** и **LangGraph**
# для построения диалоговых систем с памятью.
#
# ## Чему вы научитесь
#
# 1. Базовые принципы памяти — как сохранять и использовать контекст
# 2. Стратегии управления историей — буфер, окно, суммаризация
# 3. Управление состоянием — LangGraph StateGraph и MessagesState
# 4. Персистентность — сохранение памяти между сессиями (checkpointing)
#
# ## Установка
#
# ```bash
# uv sync
# ```

# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_mistralai import ChatMistralAI

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# %% [markdown]
# ## 1. Основы памяти: ручная передача истории сообщений
#
# В самом простом виде память — это список предыдущих сообщений,
# которые передаются языковой модели в качестве контекста.

# %%
def demonstrate_basic_memory():
    """Демонстрация сохранения истории диалога через список сообщений."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы - дружелюбный ассистент. Отвечайте кратко, опираясь на контекст разговора."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    chain = prompt | model | StrOutputParser()

    history = []

    messages = [
        "Привет! Меня зовут Алексей, и я интересуюсь искусственным интеллектом.",
        "Расскажи мне о последних достижениях в области компьютерного зрения.",
        "А какие ещё области ИИ сейчас активно развиваются?"
    ]

    for msg in messages:
        history.append(HumanMessage(content=msg))
        response = chain.invoke({"history": history, "input": msg})
        print(f"\nПользователь: {msg}")
        print(f"Ассистент: {response}")
        history.append(AIMessage(content=response))

    return history

print("\n=== 1. Основы памяти ===")
basic_memory_conversation = demonstrate_basic_memory()

# %% [markdown]
# ## 2. Стратегии управления историей
#
# При длинных диалогах история может превысить контекстное окно модели.
# Рассмотрим стратегии управления размером истории:
#
# 1. **Полный буфер** — хранит все сообщения (просто, но не масштабируется)
# 2. **Окно** — хранит только N последних сообщений
# 3. **Обрезка по токенам** — ограничивает историю по количеству токенов
# 4. **Суммаризация** — сжимает старую историю в краткую сводку

# %%
def buffer_strategy(messages):
    """Полный буфер — возвращает все сообщения."""
    return messages

def window_strategy(messages, k=3):
    """Окно — возвращает только k последних пар сообщений."""
    # k пар = 2*k сообщений
    return messages[-(k * 2):]

def token_trim_strategy(messages, max_tokens=300):
    """Обрезка по токенам — использует trim_messages из LangChain."""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=len,  # Упрощённый подсчёт (по символам)
        strategy="last",
        allow_partial=False,
    )

def summary_strategy(messages, model):
    """Суммаризация — сжимает старые сообщения в сводку."""
    if len(messages) <= 4:
        return messages

    # Берём старые сообщения для суммаризации
    old_messages = messages[:-4]
    recent_messages = messages[-4:]

    # Формируем текст для суммаризации
    old_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in old_messages
    )

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Сделайте краткую сводку диалога в 2-3 предложениях на русском."),
        ("human", "{text}")
    ])
    chain = summary_prompt | model | StrOutputParser()
    summary = chain.invoke({"text": old_text})

    return [SystemMessage(content=f"Сводка предыдущего разговора: {summary}")] + recent_messages


def demonstrate_memory_strategies():
    """Сравнение стратегий управления историей."""
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы - дружелюбный ассистент по имени Алиса. Отвечайте кратко."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = prompt | model | StrOutputParser()

    test_messages = [
        "Привет! Я Мария, мне 30 лет, я работаю инженером.",
        "Какие языки программирования подходят для AI?",
        "А какие библиотеки для глубокого обучения?",
        "Есть хорошие онлайн-курсы для начинающих?",
        "Теперь поговорим о путешествиях. Куда поехать в Европе летом?",
        "А бюджетные варианты?",
        "Напомни, как меня зовут и чем я занимаюсь?",
    ]

    strategies = {
        "Полный буфер": lambda msgs: buffer_strategy(msgs),
        "Окно (k=2)": lambda msgs: window_strategy(msgs, k=2),
        "Суммаризация": lambda msgs: summary_strategy(msgs, model),
    }

    results = {}

    for strategy_name, strategy_fn in strategies.items():
        print(f"\n{'='*50}")
        print(f"Стратегия: {strategy_name}")
        print(f"{'='*50}")

        full_history = []

        for msg in test_messages:
            full_history.append(HumanMessage(content=msg))

            # Применяем стратегию для получения контекста
            context = strategy_fn(full_history)

            response = chain.invoke({"history": context, "input": msg})
            print(f"\nПользователь: {msg}")
            print(f"Ассистент: {response}")

            full_history.append(AIMessage(content=response))

        results[strategy_name] = response

    print(f"\n{'='*50}")
    print("Сравнение ответов на последний вопрос ('Напомни, как меня зовут?'):")
    print(f"{'='*50}")
    for name, resp in results.items():
        print(f"\n{name}:\n{resp}")

    return results

print("\n=== 2. Стратегии управления историей ===")
memory_strategies_results = demonstrate_memory_strategies()

# %% [markdown]
# ### Сравнение стратегий:
#
# * **Полный буфер** — помнит всё, но может переполнить контекстное окно
# * **Окно** — помнит недавнее, но теряет ранний контекст (имя, профессию)
# * **Суммаризация** — сохраняет ключевые факты, но теряет детали

# %% [markdown]
# ## 3. Управление состоянием с помощью LangGraph
#
# LangGraph позволяет создавать диалоговые системы с управлением состоянием
# через графы. `MessagesState` автоматически управляет историей сообщений.

# %%
def demonstrate_langgraph_chatbot():
    """Чат-бот на LangGraph с MessagesState."""
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)

    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state):
        system = SystemMessage(content="Вы - дружелюбный ассистент. Отвечайте кратко.")
        messages = [system] + state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    graph = workflow.compile()

    test_messages = [
        "Привет! Меня зовут Дмитрий.",
        "Расскажи мне о машинном обучении кратко.",
        "Как меня зовут?",
    ]

    state = {"messages": []}
    for msg in test_messages:
        state["messages"].append(HumanMessage(content=msg))
        state = graph.invoke(state)
        print(f"\nПользователь: {msg}")
        print(f"Ассистент: {state['messages'][-1].content}")

    return state

print("\n=== 3. LangGraph чат-бот ===")
langgraph_state = demonstrate_langgraph_chatbot()

# %% [markdown]
# ## 4. Checkpointing — сохранение состояния между сессиями
#
# LangGraph поддерживает checkpointing для персистентности.
# Каждый пользователь получает свой `thread_id`, и состояние
# автоматически сохраняется и восстанавливается.

# %%
def demonstrate_checkpointing():
    """Демонстрация checkpointing с MemorySaver."""
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)

    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state):
        system = SystemMessage(content="Вы - ассистент с памятью. Отвечайте кратко.")
        messages = [system] + state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory_saver = MemorySaver()
    graph = workflow.compile(checkpointer=memory_saver)

    # Пользователь 1
    config1 = {"configurable": {"thread_id": "user1"}}
    state1 = {"messages": [HumanMessage(content="Привет! Меня зовут Анна.")]}
    state1 = graph.invoke(state1, config=config1)
    print(f"Пользователь 1: Привет! Меня зовут Анна.")
    print(f"Ассистент: {state1['messages'][-1].content}")

    # Пользователь 2
    config2 = {"configurable": {"thread_id": "user2"}}
    state2 = {"messages": [HumanMessage(content="Здравствуйте! Я Иван.")]}
    state2 = graph.invoke(state2, config=config2)
    print(f"\nПользователь 2: Здравствуйте! Я Иван.")
    print(f"Ассистент: {state2['messages'][-1].content}")

    # Пользователь 1 продолжает — checkpointer восстановит контекст
    state1["messages"].append(HumanMessage(content="Как меня зовут?"))
    state1 = graph.invoke(state1, config=config1)
    print(f"\nПользователь 1: Как меня зовут?")
    print(f"Ассистент: {state1['messages'][-1].content}")

    # Пользователь 2 продолжает
    state2["messages"].append(HumanMessage(content="Как меня зовут?"))
    state2 = graph.invoke(state2, config=config2)
    print(f"\nПользователь 2: Как меня зовут?")
    print(f"Ассистент: {state2['messages'][-1].content}")

    # Проверяем snapshot
    snapshot = graph.get_state(config1)
    if snapshot:
        print(f"\nСохранённое состояние User1: {len(snapshot.values['messages'])} сообщений")

    return memory_saver

print("\n=== 4. Checkpointing ===")
checkpointer = demonstrate_checkpointing()

# %% [markdown]
# ## Выводы
#
# После выполнения лабораторной работы ответьте на вопросы:
#
# 1. Какая стратегия управления историей лучше всего сохраняет контекст?
# 2. В каких случаях суммаризация предпочтительнее оконной стратегии?
# 3. Как checkpointing помогает в многопользовательских системах?
# 4. Какие хранилища состояния можно использовать в production (Redis, PostgreSQL)?
#
# ---
#
# ## Ваши наблюдения
#
# *Запишите здесь свои выводы*

# %%
# Место для дополнительных экспериментов
