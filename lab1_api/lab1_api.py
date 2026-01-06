# %% [markdown]
# # Лабораторная работа 1: Работа с API языковых моделей (Mistral AI)
#
# ## Цель
#
# Освоить работу с API крупных языковых моделей на примере Mistral AI,
# изучить параметры генерации и их влияние на результаты.
#
# ## Задачи
#
# 1. Зарегистрироваться на [Mistral AI](https://console.mistral.ai/) и получить API ключ
# 2. Настроить окружение и установить зависимости
# 3. Реализовать базовый запрос к API (chat completion)
# 4. Исследовать влияние параметров генерации:
#    - `temperature` — управление "креативностью" модели
#    - `max_tokens` — ограничение длины ответа
#    - `top_p` — nucleus sampling
# 5. Провести эксперименты и задокументировать наблюдения
#
# ## Установка
#
# ```bash
# pip install mistralai python-dotenv
# export MISTRAL_API_KEY="your_api_key"
# ```

# %% [markdown]
# ## Подготовка: Импорт библиотек и инициализация

# %%
import os
from dotenv import load_dotenv
from mistralai import Mistral

# Загрузка переменных окружения из .env файла
load_dotenv()

def get_client() -> Mistral:
    """Создание клиента Mistral API."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY не установлен. "
            "Установите: export MISTRAL_API_KEY='your_key'"
        )
    return Mistral(api_key=api_key)

# Инициализация клиента
client = get_client()
print("Клиент Mistral API инициализирован!")

# %% [markdown]
# ## 1. Базовый запрос к API (Chat Completion)
#
# Самый простой способ взаимодействия с моделью — отправка сообщения и получение ответа.
#
# **Ключевые понятия:**
# - `model` — название модели (mistral-small, mistral-medium, mistral-large)
# - `messages` — список сообщений в формате `[{"role": "user/assistant/system", "content": "текст"}]`

# %%
def basic_chat(prompt: str, model: str = "mistral-small-latest") -> str:
    """Базовый запрос к Mistral API."""
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Тестовый запрос
response = basic_chat("Привет! Расскажи о себе в двух предложениях.")
print("Ответ модели:")
print(response)

# %% [markdown]
# ## 2. Параметры генерации
#
# ### Temperature (0.0 - 2.0)
#
# Контролирует "креативность" модели:
# - **0.0 - 0.3**: Детерминированные, точные ответы. Для фактов, кода, точных задач.
# - **0.5 - 0.7**: Баланс точности и креативности. Универсальное значение.
# - **0.8 - 1.5**: Креативные, разнообразные ответы. Для генерации идей, историй.
#
# ### Max Tokens
#
# Ограничивает длину ответа. Влияет на детализацию и стоимость запроса.
#
# ### Top P (0.0 - 1.0)
#
# Nucleus sampling — отсекает маловероятные токены:
# - **0.1**: Только самые вероятные токены (фокусированный ответ)
# - **0.9-1.0**: Все токены учитываются (разнообразный ответ)

# %%
def chat_with_params(
    prompt: str,
    model: str = "mistral-small-latest",
    temperature: float = 0.7,
    max_tokens: int = 256,
    top_p: float = 1.0
) -> str:
    """Запрос с настраиваемыми параметрами генерации."""
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message.content

# %% [markdown]
# ## 3. Эксперимент: Влияние Temperature
#
# Посмотрим, как temperature влияет на креативность ответов.

# %%
prompt = "Придумай название для стартапа по доставке еды"

print("=" * 60)
print("ЭКСПЕРИМЕНТ: Влияние температуры")
print(f"Промпт: {prompt}")
print("=" * 60)

for temp in [0.0, 0.5, 1.0, 1.5]:
    print(f"\n--- Temperature: {temp} ---")
    response = chat_with_params(prompt, temperature=temp, max_tokens=100)
    print(response)

# %% [markdown]
# ## 4. Эксперимент: Влияние Max Tokens
#
# Как ограничение длины влияет на детализацию ответа.

# %%
prompt = "Расскажи историю о космическом путешествии"

print("=" * 60)
print("ЭКСПЕРИМЕНТ: Влияние max_tokens")
print(f"Промпт: {prompt}")
print("=" * 60)

for max_tokens in [50, 100, 200]:
    print(f"\n--- Max Tokens: {max_tokens} ---")
    response = chat_with_params(prompt, max_tokens=max_tokens, temperature=0.7)
    print(response)
    print(f"[Длина: {len(response)} символов]")

# %% [markdown]
# ## 5. Эксперимент: Влияние Top P
#
# Nucleus sampling: как сужение выбора токенов влияет на ответ.

# %%
prompt = "Напиши короткое стихотворение о программировании"

print("=" * 60)
print("ЭКСПЕРИМЕНТ: Влияние top_p")
print(f"Промпт: {prompt}")
print("=" * 60)

for top_p in [0.1, 0.5, 0.9, 1.0]:
    print(f"\n--- Top P: {top_p} ---")
    response = chat_with_params(prompt, top_p=top_p, temperature=0.8, max_tokens=100)
    print(response)

# %% [markdown]
# ## 6. System Prompt: Задание роли модели
#
# System prompt позволяет задать контекст, роль или стиль ответов модели.

# %%
def chat_with_system(
    system_prompt: str,
    user_prompt: str,
    model: str = "mistral-small-latest",
    temperature: float = 0.7
) -> str:
    """Запрос с системным промптом."""
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

# %% [markdown]
# ## 7. Эксперимент: Влияние System Prompt
#
# Как системный промпт меняет стиль ответа на один и тот же вопрос.

# %%
user_prompt = "Объясни что такое машинное обучение"

system_prompts = [
    ("Без системного промпта", None),
    ("Эксперт", "Ты эксперт по ML. Отвечай кратко и технически точно."),
    ("Учитель для детей", "Ты учитель, объясняющий сложные вещи простыми словами для детей 10 лет."),
    ("Поэт", "Ты поэт. Отвечай в стихотворной форме."),
]

print("=" * 60)
print("ЭКСПЕРИМЕНТ: Влияние системного промпта")
print(f"Вопрос: {user_prompt}")
print("=" * 60)

for name, system_prompt in system_prompts:
    print(f"\n--- {name} ---")
    if system_prompt:
        response = chat_with_system(system_prompt, user_prompt)
    else:
        response = chat_with_params(user_prompt, max_tokens=200)
    print(response)

# %% [markdown]
# ## Выводы
#
# После выполнения экспериментов опишите свои наблюдения:
#
# 1. **Temperature**: Как менялся характер ответов при разных значениях?
# 2. **Max Tokens**: Как ограничение длины влияло на полноту ответа?
# 3. **Top P**: Заметили ли разницу между 0.1 и 1.0?
# 4. **System Prompt**: Какой стиль был наиболее подходящим для разных задач?
#
# ---
#
# ## Ваши наблюдения
#
# *Запишите здесь свои выводы по каждому эксперименту*

# %%
# Место для ваших дополнительных экспериментов
