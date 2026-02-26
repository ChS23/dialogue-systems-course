# %% [markdown]
# # Лабораторная работа 4: Пайплайн обработки PDF и векторный поиск
#
# ## Цель
#
# Научиться создавать пайплайн для обработки PDF-документов:
# извлечение текста, разбиение на чанки, создание эмбеддингов и векторный поиск.
#
# ## Задачи
#
# 1. Конвертировать PDF в текст с помощью docling
# 2. Разбить текст на чанки (chunks) с учётом токенов
# 3. Создать эмбеддинги через Mistral API
# 4. Сохранить в векторную БД LanceDB
# 5. Реализовать семантический поиск
#
# ## Установка
#
# ```bash
# uv add mistralai docling lancedb pyarrow tiktoken tenacity python-dotenv
# ```

# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
import os
import glob
import time
import random
from typing import List, Dict, Tuple

import pyarrow as pa
from tiktoken import get_encoding
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

import lancedb
from mistralai import Mistral

# Загружаем переменные окружения
load_dotenv()

# %% [markdown]
# ## Инициализация клиента Mistral

# %%
def get_mistral_client() -> Mistral:
    """Создаёт клиент Mistral API."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY не установлен")
    return Mistral(api_key=api_key)

client = get_mistral_client()
print("Клиент Mistral API инициализирован!")

# %% [markdown]
# ## Токенизатор для HybridChunker
#
# Docling требует токенизатор для правильного разбиения текста.
# Создаём обёртку над tiktoken.

# %%
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class TiktokenWrapper(PreTrainedTokenizerBase):
    """Обёртка tiktoken для совместимости с HybridChunker."""

    def __init__(self, model_name: str = "cl100k_base", max_length: int = 8191, **kwargs):
        super().__init__(model_max_length=max_length, **kwargs)
        self.tokenizer = get_encoding(model_name)
        self._vocab_size = self.tokenizer.max_token_value

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return [str(t) for t in self.tokenizer.encode(text)]

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def get_vocab(self) -> Dict[str, int]:
        return dict(enumerate(range(self.vocab_size)))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def save_vocabulary(self, *args) -> Tuple[str]:
        return ()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

tokenizer = TiktokenWrapper()
MAX_TOKENS = 8191

print(f"Токенизатор инициализирован, max_tokens={MAX_TOKENS}")

# %% [markdown]
# ## 1. Конвертация PDF в документ
#
# Используем docling для извлечения структурированного текста из PDF.

# %%
def convert_pdf_to_document(pdf_path: str):
    """
    Конвертирует PDF в формат документа docling.

    Args:
        pdf_path: путь к PDF-файлу

    Returns:
        Объект docling с конвертированным документом
    """
    converter = DocumentConverter()
    if not pdf_path.startswith('http'):
        pdf_path = os.path.abspath(pdf_path)
    result = converter.convert(pdf_path)
    return result

# %% [markdown]
# ## 2. Обработка PDF-документов и разбиение на чанки

# %%
def process_pdf_documents(pdf_dir: str) -> list:
    """
    Обрабатывает все PDF в директории и разбивает на чанки.

    Args:
        pdf_dir: путь к директории с PDF

    Returns:
        Список чанков из всех документов
    """
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    if not pdf_files:
        print(f"В директории {pdf_dir} не найдено PDF-файлов")
        return []

    all_chunks = []

    for pdf_file in pdf_files:
        print(f"Обработка: {pdf_file}")

        result = convert_pdf_to_document(pdf_file)

        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            merge_peers=True,
        )

        chunks = list(chunker.chunk(dl_doc=result.document))
        print(f"  Извлечено {len(chunks)} чанков")
        all_chunks.extend(chunks)

    print(f"Всего: {len(all_chunks)} чанков из {len(pdf_files)} документов")
    return all_chunks

# %% [markdown]
# ## 3. Создание эмбеддингов через Mistral API
#
# Используем модель `mistral-embed` (размерность 1024).

# %%
EMBEDDING_DIM = 1024  # Размерность эмбеддингов Mistral

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(Exception)
)
def create_embedding(text: str) -> list:
    """
    Создаёт эмбеддинг для текста через Mistral API.

    Args:
        text: текст для эмбеддинга

    Returns:
        Вектор эмбеддинга (1024 dim)
    """
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding

# Тест
test_emb = create_embedding("Тестовый текст")
print(f"Эмбеддинг создан, размерность: {len(test_emb)}")

# %% [markdown]
# ## 4. Работа с LanceDB
#
# LanceDB — быстрая векторная БД для хранения и поиска эмбеддингов.

# %%
def create_or_connect_db(db_path: str = "data/lancedb"):
    """Создаёт или подключается к базе данных LanceDB."""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    return lancedb.connect(db_path)


def create_and_fill_table(db, chunks: list, table_name: str = "pdf_chunks"):
    """
    Создаёт таблицу в LanceDB и заполняет её чанками с эмбеддингами.

    Args:
        db: соединение с базой данных
        chunks: список чанков
        table_name: имя таблицы

    Returns:
        Объект таблицы LanceDB
    """
    print(f"Создание эмбеддингов для {len(chunks)} чанков...")

    # Схема таблицы
    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        pa.field("doc_name", pa.string()),
        pa.field("chunk_id", pa.int32())
    ])

    table = db.create_table(table_name, schema=schema, mode="overwrite")

    # Подготовка чанков
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
        doc_name = "unknown"
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            doc_name = chunk.metadata.get("file_name", f"doc_{i // 2}")

        processed_chunks.append({
            "text": chunk_text,
            "doc_name": doc_name,
            "chunk_id": i
        })

    # Создание эмбеддингов и добавление в таблицу
    successful = 0
    for i, chunk in enumerate(processed_chunks):
        preview = chunk["text"][:50].replace("\n", " ") + "..."
        print(f"[{i+1}/{len(processed_chunks)}] {preview}")

        try:
            vector = create_embedding(chunk["text"])
            chunk_to_add = chunk.copy()
            chunk_to_add["vector"] = vector
            table.add([chunk_to_add])
            successful += 1
            time.sleep(random.uniform(0.3, 0.8))  # Rate limit
        except Exception as e:
            print(f"  Ошибка: {e}")

    print(f"Добавлено {successful}/{len(processed_chunks)} чанков в таблицу {table_name}")
    return table

# %% [markdown]
# ## 5. Семантический поиск

# %%
def search_in_table(query_text: str, table, limit: int = 3):
    """
    Семантический поиск в таблице LanceDB.

    Args:
        query_text: текстовый запрос
        table: таблица LanceDB
        limit: количество результатов

    Returns:
        pandas.DataFrame с результатами
    """
    print(f"Поиск: '{query_text}'")

    query_embedding = create_embedding(query_text)
    results = table.search(query_embedding).limit(limit).to_pandas()

    print(f"Найдено {len(results)} результатов")
    return results


def display_search_results(results):
    """Отображает результаты поиска."""
    if results is None or len(results) == 0:
        print("Ничего не найдено")
        return

    print(f"\nНайдено {len(results)} результатов:\n")

    for i, row in results.iterrows():
        doc_name = row.get("doc_name", "unknown")
        chunk_id = row.get("chunk_id", i)
        distance = row.get("_distance", 0)
        text_preview = row['text'][:300].replace("\n", " ")

        print(f"--- Результат #{i+1} ---")
        print(f"Релевантность: {1 - distance:.4f}")
        print(f"Источник: {doc_name}, чанк #{chunk_id}")
        print(f"Текст: {text_preview}...")
        print()

# %% [markdown]
# ## 6. Полный пайплайн

# %%
def run_pipeline(pdf_dir: str, db_path: str = "data/lancedb", table_name: str = "pdf_docs"):
    """
    Запускает полный цикл обработки PDF → векторная БД.

    Args:
        pdf_dir: путь к директории с PDF
        db_path: путь к базе данных
        table_name: имя таблицы
    """
    print("=" * 50)
    print(f"PDF директория: {pdf_dir}")
    print(f"База данных: {db_path}")
    print(f"Таблица: {table_name}")
    print("=" * 50)

    # Шаг 1: Обработка PDF
    chunks = process_pdf_documents(pdf_dir)
    if not chunks:
        print("Не удалось извлечь чанки")
        return None

    # Шаг 2: Создание векторной БД
    db = create_or_connect_db(db_path)
    table = create_and_fill_table(db, chunks, table_name)

    print("\n" + "=" * 50)
    print("Пайплайн завершён!")
    print(f"Создана БД: {db_path}")
    print(f"Таблица: {table_name}")
    print(f"Чанков: {len(chunks)}")
    print("=" * 50)

    return table

# %% [markdown]
# ## Запуск пайплайна
#
# Укажите путь к директории с PDF-файлами.

# %%
# Настройки
PDF_DIR = "../lab3_html_to_pdf/bank_data_output/pdf"
DB_PATH = "./data/lancedb"
TABLE_NAME = "pdf_docs"

# Запуск (раскомментируйте для выполнения)
# table = run_pipeline(PDF_DIR, DB_PATH, TABLE_NAME)

# %% [markdown]
# ## Интерактивный поиск
#
# После создания БД можно выполнять поиск.

# %%
def interactive_search(db_path: str = "data/lancedb", table_name: str = "pdf_docs"):
    """Интерактивный поиск по созданной БД."""
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)

    print("Введите запрос (или 'exit' для выхода):")

    while True:
        query = input("\n> ")
        if query.lower() == "exit":
            break
        if query.strip():
            results = search_in_table(query, table, limit=3)
            display_search_results(results)

# Запуск интерактивного поиска (раскомментируйте)
# interactive_search(DB_PATH, TABLE_NAME)

# %% [markdown]
# ## Выводы
#
# После выполнения лабораторной работы ответьте на вопросы:
#
# 1. Как размер чанка влияет на качество поиска?
# 2. Какие преимущества векторного поиска перед полнотекстовым?
# 3. Как можно улучшить качество извлечения текста из PDF?
# 4. Какие метрики можно использовать для оценки качества поиска?
#
# ---
#
# ## Ваши наблюдения
#
# *Запишите здесь свои выводы*

# %%
# Место для дополнительных экспериментов
