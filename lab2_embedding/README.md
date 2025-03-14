# Лабораторная работа 2: Embedding с использованием Ollama

## Цель

Освоить методы генерации эмбеддингов с помощью Ollama, изучить косинусное сходство и визуализировать результаты.

## Задачи

1. Установить Ollama и запустить сервис с помощью Docker Compose.
2. Выбрать и использовать модель эмбеддингов для генерации векторных представлений текстов.
3. Подготовить набор из 50 предложений и сгенерировать их эмбеддинги.
4. Визуализировать результаты в виде heatmap с помощью plotly.

## Требования

- **Docker** и **Docker Compose** для развёртывания сервиса.
- **Python 3.11** или выше.
- **Библиотеки**: `langchain-ollama`, `scikit-learn`, `plotly`, `pandas`.

## Установка и настройка

1. **Клонирование репозитория:**
   ```bash
   git clone https://github.com/ChS23/dialogue-systems-course.git
   ```

2. **Запуск Ollama через Docker Compose:**
   ```bash
   cd compose
   docker-compose up -d --build
   ```
   Это развернёт сервис Ollama, который будет доступен для обработки эмбеддингов.

3. **Создание виртуального окружения и установка зависимостей:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate    # Windows
   cd lab2_embedding
   pip install -r requirements.txt
   ```

## Оценка результатов

- Среднее значение косинусного сходства для схожих пар предложений **должно быть ≥ 0.8**.
- Если значение ниже, попробуйте:
  - Выбрать другую модель эмбеддингов.
  - Подобрать более релевантные пары предложений.
