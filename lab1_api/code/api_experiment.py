"""
Лабораторная работа 1: Работа с API языковых моделей (Mistral AI)

Цель: Освоить работу с Mistral API и изучить влияние параметров генерации.

Перед запуском:
1. Установите зависимости: pip install -r requirements.txt
2. Установите переменную окружения: export MISTRAL_API_KEY="your_key"
"""

import os
from mistralai import Mistral


def get_client() -> Mistral:
    """Создание клиента Mistral API."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY не установлен. "
            "Установите переменную окружения: export MISTRAL_API_KEY='your_key'"
        )
    return Mistral(api_key=api_key)


def basic_chat_completion(client: Mistral, prompt: str, model: str = "mistral-small-latest") -> str:
    """
    Базовый запрос к API.

    Args:
        client: Клиент Mistral API
        prompt: Текст запроса пользователя
        model: Название модели

    Returns:
        Ответ модели
    """
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def chat_with_parameters(
    client: Mistral,
    prompt: str,
    model: str = "mistral-small-latest",
    temperature: float = 0.7,
    max_tokens: int = 256,
    top_p: float = 1.0
) -> str:
    """
    Запрос с настраиваемыми параметрами генерации.

    Args:
        client: Клиент Mistral API
        prompt: Текст запроса пользователя
        model: Название модели
        temperature: Температура (0.0-2.0). Выше = более креативно
        max_tokens: Максимальное количество токенов в ответе
        top_p: Nucleus sampling (0.0-1.0). Ниже = более фокусированно

    Returns:
        Ответ модели
    """
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message.content


def chat_with_system_prompt(
    client: Mistral,
    system_prompt: str,
    user_prompt: str,
    model: str = "mistral-small-latest",
    temperature: float = 0.7
) -> str:
    """
    Запрос с системным промптом для задания контекста/роли.

    Args:
        client: Клиент Mistral API
        system_prompt: Системный промпт (задаёт роль/контекст)
        user_prompt: Запрос пользователя
        model: Название модели
        temperature: Температура генерации

    Returns:
        Ответ модели
    """
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def experiment_temperature(client: Mistral, prompt: str):
    """
    Эксперимент: влияние температуры на генерацию.

    Демонстрирует разницу между низкой и высокой температурой.
    """
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ 1: Влияние температуры")
    print("=" * 60)
    print(f"Промпт: {prompt}\n")

    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        response = chat_with_parameters(
            client, prompt, temperature=temp, max_tokens=100
        )
        print(response)


def experiment_max_tokens(client: Mistral, prompt: str):
    """
    Эксперимент: влияние max_tokens на длину ответа.
    """
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 2: Влияние max_tokens")
    print("=" * 60)
    print(f"Промпт: {prompt}\n")

    token_limits = [50, 100, 200]

    for max_tokens in token_limits:
        print(f"\n--- Max Tokens: {max_tokens} ---")
        response = chat_with_parameters(
            client, prompt, max_tokens=max_tokens, temperature=0.7
        )
        print(response)
        print(f"[Длина ответа: {len(response)} символов]")


def experiment_top_p(client: Mistral, prompt: str):
    """
    Эксперимент: влияние top_p (nucleus sampling).
    """
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 3: Влияние top_p (nucleus sampling)")
    print("=" * 60)
    print(f"Промпт: {prompt}\n")

    top_p_values = [0.1, 0.5, 0.9, 1.0]

    for top_p in top_p_values:
        print(f"\n--- Top P: {top_p} ---")
        response = chat_with_parameters(
            client, prompt, top_p=top_p, temperature=0.8, max_tokens=100
        )
        print(response)


def experiment_system_prompt(client: Mistral):
    """
    Эксперимент: влияние системного промпта на стиль ответа.
    """
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ 4: Влияние системного промпта")
    print("=" * 60)

    user_prompt = "Объясни что такое машинное обучение"

    system_prompts = [
        ("Без системного промпта", None),
        ("Эксперт", "Ты эксперт по машинному обучению. Отвечай кратко и технически точно."),
        ("Учитель для детей", "Ты учитель, объясняющий сложные вещи простыми словами для детей 10 лет."),
        ("Поэт", "Ты поэт. Отвечай на вопросы в стихотворной форме."),
    ]

    for name, system_prompt in system_prompts:
        print(f"\n--- {name} ---")
        if system_prompt:
            response = chat_with_system_prompt(
                client, system_prompt, user_prompt, temperature=0.7
            )
        else:
            response = chat_with_parameters(
                client, user_prompt, temperature=0.7, max_tokens=200
            )
        print(response)


def main():
    """Основная функция для запуска экспериментов."""
    print("Лабораторная работа 1: Работа с Mistral API")
    print("=" * 60)

    # Инициализация клиента
    client = get_client()
    print("Клиент Mistral API инициализирован успешно!\n")

    # Базовый тест
    print("Тест базового запроса:")
    response = basic_chat_completion(client, "Привет! Как дела?")
    print(f"Ответ: {response}\n")

    # Эксперименты
    experiment_temperature(client, "Придумай название для стартапа по доставке еды")
    experiment_max_tokens(client, "Расскажи историю о космическом путешествии")
    experiment_top_p(client, "Напиши короткое стихотворение о программировании")
    experiment_system_prompt(client)

    print("\n" + "=" * 60)
    print("Эксперименты завершены!")
    print("=" * 60)


if __name__ == "__main__":
    main()
