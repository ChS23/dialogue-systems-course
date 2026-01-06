# %% [markdown]
# # Лабораторная работа 2: Embedding и анализ семантической близости
#
# ## Цель
#
# Освоить методы генерации эмбеддингов с помощью Mistral AI,
# изучить косинусное сходство и визуализировать результаты.
#
# ## Задачи
#
# 1. Использовать Mistral Embeddings API для генерации векторных представлений текстов
# 2. Подготовить набор из 50 предложений выбранной предметной области
# 3. Рассчитать косинусное сходство для пар предложений
# 4. Визуализировать результаты в виде heatmap
#
# ## Метрики
#
# - Среднее косинусное сходство для схожих предложений должно быть ≥ 0.8
#
# ## Установка
#
# ```bash
# pip install mistralai scikit-learn plotly pandas python-dotenv
# export MISTRAL_API_KEY="your_api_key"
# ```

# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral

# Загрузка переменных окружения
load_dotenv()

def get_client() -> Mistral:
    """Создание клиента Mistral API."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY не установлен")
    return Mistral(api_key=api_key)

client = get_client()
print("Клиент Mistral API инициализирован!")

# %% [markdown]
# ## 1. Что такое эмбеддинги?
#
# **Эмбеддинги** — это векторные представления текста в многомерном пространстве.
# Семантически близкие тексты имеют близкие векторы.
#
# **Mistral Embeddings:**
# - Модель: `mistral-embed`
# - Размерность: 1024
# - Поддержка: многоязычность, включая русский

# %%
def get_embeddings(texts: list[str], model: str = "mistral-embed") -> list[list[float]]:
    """
    Получение эмбеддингов для списка текстов через Mistral API.

    Args:
        texts: Список текстов для эмбеддинга
        model: Название модели (mistral-embed)

    Returns:
        Список векторов эмбеддингов
    """
    response = client.embeddings.create(
        model=model,
        inputs=texts
    )
    return [item.embedding for item in response.data]

# Тест: получим эмбеддинг для одного предложения
test_embedding = get_embeddings(["Привет, мир!"])
print(f"Размерность эмбеддинга: {len(test_embedding[0])}")
print(f"Первые 5 значений: {test_embedding[0][:5]}")

# %% [markdown]
# ## 2. Подготовка данных
#
# Для лабораторной работы необходимо подготовить минимум **50 предложений**
# выбранной предметной области.
#
# Ниже приведён пример с 4 предложениями — **замените на свои данные**.

# %%
# ПРИМЕР: небольшой набор для демонстрации
# TODO: Замените на 50+ предложений вашей предметной области!

sentences = [
    "Привет, как дела?",
    "Здравствуйте, как вы поживаете?",
    "Сегодня прекрасная погода.",
    "Я люблю программировать на Python.",
    "Добрый день, как ваши дела?",
    "Python — отличный язык программирования.",
    "На улице солнечно и тепло.",
    "Как твоё самочувствие сегодня?",
]

print(f"Количество предложений: {len(sentences)}")
print("\nПримеры предложений:")
for i, s in enumerate(sentences[:5], 1):
    print(f"  {i}. {s}")

# %% [markdown]
# ## 3. Генерация эмбеддингов
#
# Получим векторные представления для всех предложений.

# %%
print("Генерация эмбеддингов...")
embeddings = get_embeddings(sentences)
print(f"Сгенерировано {len(embeddings)} эмбеддингов")
print(f"Размерность каждого: {len(embeddings[0])}")

# %% [markdown]
# ## 4. Вычисление косинусного сходства
#
# **Косинусное сходство** — мера схожести двух векторов:
# - **1.0** — полностью идентичны
# - **0.0** — ортогональны (не связаны)
# - **-1.0** — противоположны
#
# Формула: `cos(θ) = (A · B) / (||A|| × ||B||)`

# %%
# Вычисление матрицы косинусного сходства
similarity_matrix = cosine_similarity(embeddings)

print(f"Размер матрицы: {similarity_matrix.shape}")
print("\nПример значений (первые 4x4):")
print(similarity_matrix[:4, :4].round(3))

# %% [markdown]
# ## 5. Визуализация: Тепловая карта (Heatmap)
#
# Визуализируем матрицу сходства для наглядного анализа.

# %%
# Создание heatmap
fig = px.imshow(
    similarity_matrix,
    x=sentences,
    y=sentences,
    color_continuous_scale='Viridis',
    aspect='auto',
    text_auto='.2f',
    title='Матрица косинусного сходства'
)

fig.update_layout(
    xaxis_title='Предложения',
    yaxis_title='Предложения',
    xaxis_tickangle=45,
    height=600,
    width=800
)

fig.show()

# %% [markdown]
# ## 6. Анализ схожих пар
#
# Найдём пары предложений с высоким косинусным сходством (≥ 0.8).

# %%
threshold = 0.8
n = len(sentences)
similar_pairs = []

# Перебираем все уникальные пары (i, j) где i < j
for i in range(n):
    for j in range(i + 1, n):
        sim = similarity_matrix[i, j]
        if sim >= threshold:
            similar_pairs.append({
                'Предложение 1': sentences[i],
                'Предложение 2': sentences[j],
                'Сходство': round(sim, 4)
            })

# Создаём DataFrame и сортируем
df_pairs = pd.DataFrame(similar_pairs)
if not df_pairs.empty:
    df_pairs = df_pairs.sort_values('Сходство', ascending=False)
    print(f"Найдено {len(df_pairs)} пар с сходством ≥ {threshold}:\n")
    print(df_pairs.to_string(index=False))
else:
    print(f"Пар с сходством ≥ {threshold} не найдено.")
    print("Попробуйте снизить порог или добавить более схожие предложения.")

# %% [markdown]
# ## 7. Статистика по матрице сходства

# %%
import numpy as np

# Извлекаем значения выше диагонали (уникальные пары)
upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]

print("Статистика косинусного сходства:")
print(f"  Минимум:  {upper_triangle.min():.4f}")
print(f"  Максимум: {upper_triangle.max():.4f}")
print(f"  Среднее:  {upper_triangle.mean():.4f}")
print(f"  Медиана:  {np.median(upper_triangle):.4f}")

# Среднее для пар выше порога
if len(similar_pairs) > 0:
    avg_similar = df_pairs['Сходство'].mean()
    print(f"\nСреднее для пар ≥ {threshold}: {avg_similar:.4f}")

# %% [markdown]
# ## Выводы
#
# После выполнения лабораторной работы ответьте на вопросы:
#
# 1. Какие пары предложений показали наибольшее сходство? Почему?
# 2. Удалось ли достичь среднего сходства ≥ 0.8 для схожих пар?
# 3. Как модель справилась с синонимами и перефразированиями?
# 4. Какие ограничения эмбеддингов вы заметили?
#
# ---
#
# ## Ваши наблюдения
#
# *Запишите здесь свои выводы*

# %%
# Место для дополнительных экспериментов
