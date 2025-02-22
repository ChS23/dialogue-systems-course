# Лабораторная работа 2: Embedding с использованием Ollama

## Цель

Освоить методы генерации эмбеддингов с помощью Ollama, изучить косинусное сходство и визуализировать результаты.

## Задачи

1. Установить Ollama и запустить сервис с помощью Docker Compose.
2. Выбрать и использовать модель эмбеддингов для генерации векторных представлений текстов.
3. Подготовить набор из 50 предложений и сгенерировать их эмбеддинги.
4. Рассчитать косинусное сходство для 20 пар предложений.
5. Визуализировать результаты в виде heatmap с помощью matplotlib или plotly.

## Требования

- **Docker** и **Docker Compose** для развёртывания сервиса.
- **Python 3.11** или выше.
- **Библиотеки**: `langchain-ollama`, `scikit-learn`, `matplotlib`, `plotly`, `numpy`, `pandas`.

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
   pip install -r requirements.txt
   ```

## Код для генерации эмбеддингов и расчёта сходства

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import pandas as pd
from langchain_ollama.embeddings import OllamaEmbeddings

# Инициализация модели эмбеддингов
embeddings_model = OllamaEmbeddings(model="phi4")  # Можно заменить на другую модель

# Пример списка предложений
sentences = [
    "Первое предложение.",
    "Второе предложение, похожее на первое.",
    # Добавьте минимум 50 предложений
]

# Генерация эмбеддингов
sentence_embeddings = [embeddings_model.embed_query(sentence) for sentence in sentences]

# Преобразование в массив
embeddings_array = np.array(sentence_embeddings)

# Расчёт матрицы косинусного сходства
similarity_matrix = cosine_similarity(embeddings_array)

# Визуализация heatmap
df_sim = pd.DataFrame(similarity_matrix)
fig = px.imshow(df_sim, labels=dict(x="Предложение", y="Предложение", color="Сходство"),
                x=df_sim.columns, y=df_sim.index, color_continuous_scale='Viridis')
fig.update_xaxes(side="top")
fig.show()
```

## Оценка результатов

- Среднее значение косинусного сходства для схожих пар предложений **должно быть ≥ 0.8**.
- Если значение ниже, попробуйте:
  - Выбрать другую модель эмбеддингов.
  - Подобрать более релевантные пары предложений.

## Отчётность

1. **Генерируемые эмбеддинги**: Сохраните их в `embeddings.json`.
2. **Матрица сходства**: Сохраните heatmap как `similarity_heatmap.png`.
3. **Выводы**: Опишите в `report.md` влияние модели эмбеддингов на качество результатов.

## Заключение

Эта лабораторная работа знакомит с основами генерации эмбеддингов и их применением для анализа семантической близости текстов. Использование Ollama в контейнере упрощает работу с LLM и позволяет применять его в различных NLP-задачах.
