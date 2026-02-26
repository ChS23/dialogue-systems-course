# Интеграция Langfuse для трассировки LangChain / LangGraph

[Langfuse](https://langfuse.com/) — open-source платформа для наблюдаемости (observability) LLM-приложений.
Позволяет отслеживать вызовы моделей, задержки, стоимость и качество ответов.

## Зачем

- Видеть полный trace каждого запроса: какие инструменты вызвал агент, сколько токенов потрачено
- Отслеживать задержки и узкие места в пайплайне
- Сравнивать качество ответов между версиями промптов
- Группировать запросы по пользователям и сессиям

## Регистрация

1. Создайте аккаунт на [cloud.langfuse.com](https://cloud.langfuse.com/)
2. Создайте проект
3. Перейдите в **Settings → API Keys** и скопируйте `Public Key` и `Secret Key`
4. Добавьте в `.env` в корне проекта:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Установка

```bash
uv add langfuse
```

## Использование с LangChain

Интеграция работает через `CallbackHandler` — достаточно передать его в `config`.

Документация: [langfuse.com/docs/integrations/langchain/tracing](https://langfuse.com/docs/integrations/langchain/tracing)

```python
from langfuse.langchain import CallbackHandler
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Создаём handler (читает ключи из переменных окружения)
langfuse_handler = CallbackHandler()

# Обычная цепочка
prompt = ChatPromptTemplate.from_template("Объясни что такое {topic}")
model = ChatMistralAI(model="mistral-small-latest", temperature=0)
chain = prompt | model | StrOutputParser()

# Добавляем трассировку через config
response = chain.invoke(
    {"topic": "векторные базы данных"},
    config={"callbacks": [langfuse_handler]}
)
```

После выполнения trace появится в дашборде Langfuse с деталями:
- Входной промпт и ответ модели
- Количество токенов (input/output)
- Задержка вызова
- Стоимость

## Использование с LangGraph

Тот же `CallbackHandler` работает с LangGraph `StateGraph`.

Документация: [langfuse.com/docs/integrations/langchain/example-python-langgraph](https://langfuse.com/docs/integrations/langchain/example-python-langgraph)

```python
from langfuse.langchain import CallbackHandler
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, MessagesState

langfuse_handler = CallbackHandler()

model = ChatMistralAI(model="mistral-small-latest", temperature=0)

# Создаём граф
workflow = StateGraph(state_schema=MessagesState)

def call_model(state):
    system = SystemMessage(content="Вы - дружелюбный ассистент.")
    messages = [system] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
graph = workflow.compile()

# Вызов с трассировкой
response = graph.invoke(
    {"messages": [HumanMessage(content="Привет!")]},
    config={"callbacks": [langfuse_handler]}
)
```

## Трассировка агента с инструментами

Langfuse автоматически отслеживает вызовы инструментов внутри агентов.

```python
from langfuse.langchain import CallbackHandler
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain_core.tools import tool

langfuse_handler = CallbackHandler()

@tool
def get_weather(location: str) -> str:
    """Получает погоду в указанном месте."""
    return f"В {location} солнечно, 20°C"

llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

agent = create_agent(
    llm,
    [get_weather],
    system_prompt="Вы - информационный ассистент."
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Какая погода в Москве?"}]},
    config={"callbacks": [langfuse_handler]}
)
```

В trace будет видно:
- Решение агента вызвать инструмент `get_weather`
- Входные аргументы и результат инструмента
- Финальный ответ модели

## Метаданные: пользователь, сессия, теги

```python
response = chain.invoke(
    {"topic": "LangChain"},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": "user-123",
            "langfuse_session_id": "session-456",
            "langfuse_tags": ["lab5", "experiment"]
        }
    }
)
```

## Оценка качества (Scoring)

```python
from langfuse import get_client

langfuse = get_client()

langfuse.create_score(
    trace_id="trace-id-from-dashboard",
    name="user-feedback",
    value=1,
    data_type="NUMERIC",
    comment="Ответ корректный"
)
```

## Ссылки

- [Langfuse Cloud](https://cloud.langfuse.com/) — дашборд
- [Документация LangChain интеграции](https://langfuse.com/docs/integrations/langchain/tracing)
- [Примеры LangGraph + Langfuse](https://langfuse.com/docs/integrations/langchain/example-python-langgraph)
- [GitHub Langfuse](https://github.com/langfuse/langfuse)
- [Self-hosting Langfuse](https://langfuse.com/docs/deployment/self-host)
