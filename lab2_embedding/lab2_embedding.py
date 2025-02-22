import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama.embeddings import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Пример небольшого набора предложений
sentences = [
    "Привет, как дела?",
    "Здравствуйте, как вы поживаете?",
    "Сегодня прекрасная погода.",
    "Я люблю программировать на Python."
]

# Для лабораторной работы: дополните список не менее чем 50 предложениями.

# Генерация эмбеддингов для предложений
embeddings = embeddings_model.embed_documents(sentences)

# Вычисление матрицы косинусного сходства
similarity_matrix = cosine_similarity(embeddings)

# Создание аннотированной тепловой карты
fig = px.imshow(
    similarity_matrix,
    x=sentences,
    y=sentences,
    color_continuous_scale='Viridis',
    aspect='auto',
    text_auto='.2f'
)

# Обновление макета графика
fig.update_layout(
    title='Матрица косинусного сходства',
    xaxis_title='Предложения',
    yaxis_title='Предложения'
)

# Отображение графика
fig.show()

# Устанавливаем пороговое значение сходства
threshold = 0.8
n = len(sentences)
similar_pairs = []

# Перебираем все уникальные пары (i, j) с i < j
for i in range(n):
    for j in range(i + 1, n):
        sim = similarity_matrix[i, j]
        if sim >= threshold:
            similar_pairs.append((sentences[i], sentences[j], sim))

# Преобразуем список пар в DataFrame и сортируем по убыванию сходства
df_pairs = pd.DataFrame(similar_pairs, columns=['Предложение 1', 'Предложение 2', 'Сходство'])
df_pairs = df_pairs.sort_values(by='Сходство', ascending=False).head(20)

# Выводим таблицу с результатами
df_pairs