import os
import glob
from typing import List, Dict, Any, Optional, Tuple
import time
import random
import numpy as np
import pyarrow as pa

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

from tiktoken import get_encoding

# Необходимая библиотека для токенизатора 
import transformers

# Импортируем компоненты из docling
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

# Для работы с эмбеддингами и векторной БД
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

# Для работы с OpenAI
from openai import OpenAI
import openai

# Для загрузки переменных окружения
from dotenv import load_dotenv

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown, HTML

# Загружаем переменные окружения из .env файла
load_dotenv()

# Реализация токенизатора для работы с OpenAI
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Создаем класс-обертку для токенизатора OpenAI, совместимый с интерфейсом HybridChunker
class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
    """Минимальная обертка для токенизатора OpenAI."""

    def __init__(
        self, model_name: str = "cl100k_base", max_length: int = 8191, **kwargs
    ):
        """Инициализация токенизатора.

        Args:
            model_name: Название кодировки OpenAI для использования
            max_length: Максимальная длина последовательности
        """
        super().__init__(model_max_length=max_length, **kwargs)
        self.tokenizer = get_encoding(model_name)
        self._vocab_size = self.tokenizer.max_token_value

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Основной метод, используемый HybridChunker."""
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
        """Классовый метод для соответствия интерфейсу HuggingFace."""
        return cls()
    
client = OpenAI(api_key="")

# Создаем экземпляр токенизатора для OpenAI
tokenizer = OpenAITokenizerWrapper()

# Максимальное количество токенов для модели text-embedding-3-large
MAX_TOKENS = 8191

def convert_pdf_to_document(pdf_path):
    """
    Конвертирует PDF в формат документа docling.
    
    Args:
        pdf_path: путь к PDF-файлу
        
    Returns:
        Объект docling с конвертированным документом
    """
    converter = DocumentConverter()
    # # Для локальных файлов нужно использовать file:// протокол
    if not pdf_path.startswith('http'):
         pdf_path = f"{os.path.abspath(pdf_path)}"
    result = converter.convert(pdf_path)
    return result


def process_pdf_documents(pdf_dir):
    """
    Обрабатывает все PDF-документы в указанной директории.
    
    Args:
        pdf_dir: путь к директории с PDF-файлами
        
    Returns:
        Список чанков из всех документов
    """
    # Получаем все PDF файлы в директории
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"В директории {pdf_dir} не найдено PDF-файлов")
        return []
    
    all_chunks = []
    
    # Обрабатываем каждый PDF файл
    for pdf_file in pdf_files:
        print(f"Обработка файла: {pdf_file}")
        
        # Конвертируем PDF в формат docling
        result = convert_pdf_to_document(pdf_file)
        
        # Создаем чанкер
        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            merge_peers=True,  # Объединяем соседние чанки при возможности
        )
        
        # Разбиваем документ на чанки
        chunk_iter = chunker.chunk(dl_doc=result.document)
        chunks = list(chunk_iter)
        
        print(f"Извлечено {len(chunks)} чанков из документа {pdf_file}")
        all_chunks.extend(chunks)
    
    print(f"Всего извлечено {len(all_chunks)} чанков из {len(pdf_files)} документов")
    return all_chunks

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.APIConnectionError))
)
def create_embedding(text):
    """
    Создает эмбеддинг для текста с использованием OpenAI API.
    
    Функция использует декоратор retry для автоматического повтора
    при ошибках API (rate limits, timeout и т.д.)
    
    Args:
        text: текст для создания эмбеддинга
        
    Returns:
        Вектор эмбеддинга
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1536
    )
    return response.data[0].embedding

def create_or_connect_db(db_path="data/lancedb"):
    """
    Создает или подключается к базе данных LanceDB.
    
    Args:
        db_path: путь к базе данных
        
    Returns:
        Объект соединения с базой данных
    """
    # Создаем директорию для базы данных, если она не существует
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Подключаемся к базе данных
    return lancedb.connect(db_path)


def create_and_fill_table(db, chunks, table_name="pdf_chunks"):
    """
    Создает таблицу в LanceDB и заполняет ее чанками с эмбеддингами.
    
    Args:
        db: соединение с базой данных
        chunks: список чанков для добавления в таблицу
        table_name: имя создаваемой таблицы
        
    Returns:
        Объект таблицы LanceDB
    """
    print(f"🧠 Начинаем создание эмбеддингов для {len(chunks)} чанков...")
    
    # Определяем схему таблицы с использованием PyArrow
    print("📋 Определяем схему таблицы...")
    schema = pa.schema([
        pa.field("text", pa.string()),                      # Текст чанка
        pa.field("vector", pa.list_(pa.float32(), 1536)),   # Эмбеддинг (вектор)
        pa.field("doc_name", pa.string()),                  # Имя документа
        pa.field("chunk_id", pa.int32())                    # ID чанка
    ])
    
    # Создаем таблицу
    print("🏗️ Создаем таблицу в базе данных...")
    table = db.create_table(table_name, schema=schema, mode="overwrite")
    
    # Подготавливаем чанки для добавления в таблицу
    print("🔄 Подготавливаем чанки для добавления...")
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Безопасное получение текста из чанка
        if hasattr(chunk, "text"):
            chunk_text = chunk.text
        else:
            chunk_text = str(chunk)
        
        # Получаем имя документа из метаданных или генерируем
        doc_name = "unknown"
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict) and "file_name" in chunk.metadata:
            doc_name = chunk.metadata["file_name"]
        else:
            doc_name = f"doc_{i // 2}"  # Простая группировка если нет метаданных
        
        processed_chunks.append({
            "text": chunk_text,
            "doc_name": doc_name,
            "chunk_id": i
        })
    

    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=len(processed_chunks),
        description='Прогресс:',
        bar_style='info',
        orientation='horizontal'
    )
    display(progress)
    
    # Счетчик успешно добавленных чанков
    successful_chunks = 0
    
    # Создаем и добавляем каждый чанк отдельно
    print(f"🚀 Начинаем создание эмбеддингов и добавление в таблицу ({len(processed_chunks)} чанков)...")
    for i, chunk in enumerate(processed_chunks):
        chunk_preview = chunk["text"][:30].replace("\n", " ") + "..."
        print(f"Обработка чанка {i+1}/{len(processed_chunks)}: '{chunk_preview}'")
        
        try:
            # Создаем эмбеддинг для текущего чанка
            print(f"   🧮 Создание эмбеддинга...")
            vector = create_embedding(chunk["text"])
            
            # Добавляем вектор к данным чанка
            chunk_to_add = chunk.copy()
            chunk_to_add["vector"] = vector
            
            # Добавляем чанк в таблицу
            print(f"   💾 Добавление в таблицу...")
            table.add([chunk_to_add])
            
            print(f"✅ Чанк {i+1} успешно обработан и добавлен в таблицу.")
            successful_chunks += 1
            
            # Обновляем прогресс
            progress.value = i + 1
            
            # Небольшая задержка между запросами для избежания rate limits
            time.sleep(random.uniform(0.5, 1.5))
            
        except Exception as e:
            print(f"❌ Ошибка при обработке чанка {i+1}: {str(e)}")
            print(f"   Тип ошибки: {type(e).__name__}")
    
    print(f"\n🎉 Готово! {successful_chunks} из {len(processed_chunks)} чанков успешно добавлены в таблицу {table_name}")
    
    return table

def search_in_table(query_text, table, limit=3):
    """
    Поиск в таблице LanceDB по текстовому запросу.
    
    Этот метод автоматически:
    1. Создает эмбеддинг для запроса 
    2. Выполняет векторный поиск ближайших соседей
    3. Возвращает наиболее релевантные результаты
    
    Args:
        query_text (str): Текстовый запрос
        table: Таблица LanceDB для поиска
        limit (int): Количество результатов
        
    Returns:
        pandas.DataFrame: Результаты поиска
    """
    print(f"🔍 Обрабатываем запрос: '{query_text}'")
    
    try:
        # Создаем эмбеддинг для запроса
        print("🧠 Создаем эмбеддинг для запроса...")
        query_embedding = create_embedding(query_text)
        print(f"✅ Эмбеддинг создан, размерность: {len(query_embedding)}")
        
        # Выполняем векторный поиск по эмбеддингу
        print(f"🔎 Ищем {limit} наиболее релевантных чанков...")
        results = table.search(query_embedding).limit(limit).to_pandas()
        
        print(f"📊 Найдено {len(results)} результатов")
        return results
        
    except Exception as e:
        print(f"❌ Ошибка при поиске: {str(e)}")
        raise


def display_search_results(results):
    """
    Отображает результаты поиска в красивом формате.
    
    Args:
        results (pandas.DataFrame): Результаты поиска из search_in_table
    """
    if results is None or len(results) == 0:
        print("❌ По вашему запросу ничего не найдено")
        return
    
    print(f"🔍 Найдено {len(results)} результатов:")
    
    for i, row in results.iterrows():
        doc_name = row.get("doc_name", "Неизвестный документ")
        chunk_id = row.get("chunk_id", i)
        distance = row.get("_distance", "н/д")
        
        # Форматируем текст для отображения
        text_preview = row['text'][:300].replace("\n", "<br>")
        
        # Создаем HTML для красивого отображения результатов
        text = f"""
            Результат #{i+1} (релевантность: {distance:.4f})\n
            Источник:   {doc_name}\n
            ID чанка:   {chunk_id}\n
            Текст:      {text_preview}... \n
        """
        print(text)


    """
Этот блок позволяет интерактивно настроить параметры и запустить 
полный цикл обработки: от загрузки PDF до создания векторной базы данных.
"""

def run_pipeline(pdf_dir=None, db_path=None, table_name=None):
    """
    Запускает полный цикл обработки: от загрузки PDF до создания векторной базы данных.
    
    Args:
        pdf_dir: Путь к директории с PDF-файлами 
        db_path: Путь для сохранения базы данных
        table_name: Имя таблицы в базе данных
    """
    # Если параметры не указаны, запрашиваем их
    if not pdf_dir:
        pdf_dir = input("Введите путь к директории с PDF-файлами: ")
    if not db_path:
        db_path = input("Введите путь для сохранения базы данных [data/lancedb]: ") or "data/lancedb"
    if not table_name:
        table_name = input("Введите имя таблицы в базе данных [pdf_docs]: ") or "pdf_docs"
    
    print("\n" + "="*50)
    print(f"🚀 Запускаем обработку PDF документов")
    print(f"📂 Директория с PDF: {pdf_dir}")
    print(f"💾 База данных: {db_path}")
    print(f"📋 Имя таблицы: {table_name}")
    print("="*50 + "\n")
    
    # Шаг 1: Обработка PDF-документов
    chunks = process_pdf_documents(pdf_dir)

    if not chunks:
        print("❌ Не удалось извлечь чанки из документов")
        return False
    
    # Шаг 2: Создание и заполнение таблицы LanceDB
    print("\n" + "="*50)
    print("💾 Создание векторной базы данных")
    print("="*50)
    
    db = create_or_connect_db(db_path)
    print("✅ Соединение с БД установлено")
    table = create_and_fill_table(db, chunks, table_name)
    
    # Шаг 3: Информация о результатах
    print("\n" + "="*50)
    print("🎉 Итоги обработки")
    print(f"📂 Создана база данных: {db_path}")
    print(f"📋 Создана таблица: {table_name}")
    print(f"📊 Всего добавлено чанков: {len(chunks)}")
    print("="*50)
    
    return True

"""
Этот блок позволяет интерактивно выполнять поиск в созданной базе данных.
"""

def interactive_search(db_path=None, table_name=None):
    """
    Запускает интерактивный поиск по созданной базе данных.
    
    Args:
        db_path: Путь к базе данных
        table_name: Имя таблицы для поиска
    """
    # Если параметры не указаны, запрашиваем их
    if not db_path:
        db_path = input("Введите путь к базе данных [data/lancedb]: ") or "data/lancedb"
    if not table_name:
        table_name = input("Введите имя таблицы в базе данных [pdf_docs]: ") or "pdf_docs"
    
    try:
        # Подключаемся к базе данных
        db = lancedb.connect(db_path)
        
        # Открываем таблицу
        table = db.open_table(table_name)
        
        # Поле ввода запроса
        query_input = widgets.Text(
            value='',
            placeholder='Введите ваш запрос...',
            description='Запрос:',
            disabled=False,
            layout=widgets.Layout(width='80%')
        )
        
        # Слайдер для выбора количества результатов
        limit_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=10,
            step=1,
            description='Кол-во результатов:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        # Кнопка поиска
        search_button = widgets.Button(
            description='Поиск',
            button_style='success',
            tooltip='Нажмите для поиска',
            icon='search'
        )
        
        # Вывод результатов
        output = widgets.Output()
        
        # Функция для обработки нажатия кнопки
        def on_search_button_click(b):
            query = query_input.value
            limit = limit_slider.value
            
            if not query:
                with output:
                    clear_output()
                    display(HTML("<p style='color:red'>⚠️ Пожалуйста, введите запрос</p>"))
                return
            
            with output:
                clear_output()
                print(f"🔍 Поиск по запросу: '{query}'")
                results = search_in_table(query, table, limit)
                display_search_results(results)
        
        # Привязываем функцию к кнопке
        search_button.on_click(on_search_button_click)
        
        # Отображаем интерфейс
        display(widgets.VBox([
            widgets.HBox([query_input, search_button]),
            limit_slider,
            output
        ]))
    
    except Exception as e:
        print(f"❌ Ошибка при подключении к базе данных: {str(e)}")

PDF_DIR = "../bank_data_output/pdf"
DB_PATH = "./lancedb"
TABLE_NAME = "pdf_docs" 

# Запускаем обработку
success = run_pipeline(PDF_DIR, DB_PATH, TABLE_NAME)

# Если обработка успешна, запускаем интерактивный поиск
if success:
    # Подключаемся к базе данных
    db = lancedb.connect(DB_PATH)
        
    # Открываем таблицу
    table = db.open_table(TABLE_NAME)
    question = ""
    while(question!="exit"):
        question = input("Enter your question or 'exit' to quit: ")
        if question != "exit":
            results = search_in_table(question, table, 3)
            display_search_results(results)
    