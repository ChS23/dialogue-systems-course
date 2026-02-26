# Курс по Диалоговым Системам

Репозиторий курса по диалоговым системам. Материалы и лабораторные работы по разработке современных диалоговых агентов.

## Описание курса

Цель курса — дать практические навыки разработки диалоговых систем, охватывая полный цикл: от работы с API языковых моделей до интеграции в Telegram-бота. В ходе курса вы научитесь:

- Работать с API языковых моделей (Mistral AI) и настраивать параметры генерации
- Генерировать эмбеддинги и анализировать семантическую близость текстов
- Извлекать и обрабатывать данные с веб-сайтов
- Создавать пайплайны обработки документов и хранить данные в векторных БД
- Использовать LangChain для создания агентов с инструментами
- Добавлять механизмы памяти и управления состоянием (LangGraph)
- Интегрировать все компоненты в Telegram-бота

Ознакомьтесь с доступными предметными областями в каталоге [`domain_selection/`](domain_selection/) и выберите направление для лабораторных работ.

## Лабораторные работы

| # | Тема | Технологии |
|---|------|------------|
| 1 | [API языковых моделей](lab1_api/) | Mistral API, параметры генерации |
| 2 | [Эмбеддинги](lab2_embedding/) | mistral-embed, cosine similarity, plotly |
| 3 | [Парсинг HTML → PDF](lab3_html_to_pdf/) | BeautifulSoup, Playwright |
| 4 | [PDF пайплайн и векторный поиск](lab4_pdf_pipeline/) | docling, LanceDB, tiktoken |
| 5 | [LangChain и AI-агенты](lab5_web_search/) | LangChain, Tools, Tavily Search |
| 6 | [Память и состояние](lab6_memory_history/) | LangGraph, MessagesState, checkpointing |
| 7 | [Поисковый агент по документам](lab7_document_search/) | LangGraph, LanceDB, RAG |
| 8 | [Telegram-бот](lab8_tg_bot/) | aiogram 3.x, uvloop, orjson |

## Требования

- **Python 3.12** или выше
- **uv** — менеджер пакетов ([установка](https://docs.astral.sh/uv/getting-started/installation/))
- **API ключи** (в файле `.env` в корне проекта):
  - `MISTRAL_API_KEY` — получить на [console.mistral.ai](https://console.mistral.ai/) (раздел API Keys)
  - `TAVILY_API_KEY` — получить на [app.tavily.com](https://app.tavily.com/) (нужен для lab5)
  - `BOT_TOKEN` — получить у [@BotFather](https://t.me/BotFather) в Telegram (нужен для lab8)

Пример файла `.env`:

```env
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key
BOT_TOKEN=your_telegram_bot_token
```

## Установка и запуск

```bash
git clone https://github.com/ChS23/dialogue-systems-course.git
cd dialogue-systems-course

# Установка зависимостей
uv sync

# Установка браузера для lab3
uv run playwright install chromium

# Запуск любой лабораторной
uv run python lab1_api/lab1_api.py
```

## Структура проекта

```
dialogue-systems-course/
├── pyproject.toml          # Все зависимости проекта
├── uv.lock                 # Зафиксированные версии
├── .env                    # API ключи (не в git)
├── lab1_api/               # Работа с Mistral API
├── lab2_embedding/         # Эмбеддинги и сходство
├── lab3_html_to_pdf/       # Парсинг веб-страниц
├── lab4_pdf_pipeline/      # PDF → чанки → LanceDB
├── lab5_web_search/        # LangChain, агенты, веб-поиск
├── lab6_memory_history/    # Память и управление состоянием
├── lab7_document_search/   # RAG-агент по документам
├── lab8_tg_bot/            # Telegram-бот (курсовая)
├── domain_selection/       # Выбор предметной области
└── compose/                # Docker Compose (Ollama)
```
