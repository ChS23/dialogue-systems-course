# %% [markdown]
# # Лабораторная работа 3: Извлечение данных с веб-сайтов и конвертация в PDF
#
# ## Цель
#
# Научиться извлекать данные с веб-сайтов, обрабатывать HTML и конвертировать в PDF.
#
# ## Задачи
#
# 1. Получить список URL из sitemap.xml
# 2. Загрузить HTML-страницы с корректной обработкой кодировки
# 3. Определить информативность страницы (фильтрация)
# 4. Извлечь основной контент с помощью BeautifulSoup
# 5. Конвертировать HTML в PDF с помощью Playwright
#
# ## Установка
#
# ```bash
# pip install requests beautifulsoup4 lxml playwright pyhtml2pdf
# playwright install chromium
# ```

# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
import os
import re
import time
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# %% [markdown]
# ## Настройка путей

# %%
OUTPUT_DIRECTORY = 'bank_data_output'
HTML_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'html')
PDF_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'pdf')
TEXT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'text')

# %% [markdown]
# ## 1. Создание директорий для сохранения данных

# %%
def create_output_folders():
    """Создаёт все необходимые директории для сохранения файлов."""
    for directory in [OUTPUT_DIRECTORY, HTML_DIRECTORY, PDF_DIRECTORY, TEXT_DIRECTORY]:
        os.makedirs(directory, exist_ok=True)
    print("Директории для сохранения файлов созданы успешно")

# Создаём директории
create_output_folders()

# %% [markdown]
# ## 2. Получение списка URL из карты сайта (sitemap.xml)
#
# Sitemap — это XML-файл, содержащий список всех страниц сайта.
# Используется поисковыми системами и для автоматизации сбора данных.

# %%
def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """
    Получает список URL из sitemap.xml.

    Args:
        sitemap_url: URL карты сайта

    Returns:
        Список URL из карты сайта
    """
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        # Используем lxml парсер для XML
        sitemap_soup = BeautifulSoup(response.content, 'lxml-xml')
        # Извлекаем все URL из тега <loc>
        urls = [loc.text for loc in sitemap_soup.find_all('loc')]
        print(f'Найдено {len(urls)} URL из карты сайта.')
        return urls
    except Exception as e:
        print(f'Ошибка при загрузке карты сайта: {e}')
        # Резервный список URL
        backup_urls = [
            "https://www.tbank.ru/business/",
            "https://www.tbank.ru/business/help/",
            "https://www.tbank.ru/business/cards/"
        ]
        print(f'Используем резервный список из {len(backup_urls)} URL')
        return backup_urls

# %% [markdown]
# ## 3. Загрузка HTML-страниц с обработкой кодировки
#
# Важно корректно определить кодировку страницы (UTF-8, Windows-1251 и т.д.)
# для правильного отображения кириллицы.

# %%
def download_webpage(url: str) -> tuple[str, str | None]:
    """
    Загружает HTML-страницу и корректно обрабатывает кодировку.

    Args:
        url: URL страницы для загрузки

    Returns:
        Кортеж (имя_файла, содержимое_html) или (имя_файла, None) при ошибке
    """
    print(f"Загрузка страницы: {url}")

    try:
        # Заголовки для имитации браузера
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru,en-US;q=0.7,en;q=0.3',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Определяем кодировку
        if response.encoding.upper() == 'ISO-8859-1':
            encoding = None
            # Проверяем в заголовке Content-Type
            content_type = response.headers.get('Content-Type', '')
            charset_match = re.search(r'charset=([\w-]+)', content_type)
            if charset_match:
                encoding = charset_match.group(1)

            if not encoding:
                # Ищем в meta-теге HTML
                meta_match = re.search(
                    r'<meta[^>]*charset=["\']*([^\"\'>]+)',
                    response.text,
                    re.IGNORECASE
                )
                if meta_match:
                    encoding = meta_match.group(1)

            response.encoding = encoding or 'utf-8'

        # Генерируем имя файла из URL
        url_parts = urlparse(url)
        path = url_parts.path.rstrip('/')
        if path:
            filename = os.path.basename(path) or url_parts.netloc.replace('.', '_')
        else:
            filename = url_parts.netloc.replace('.', '_')

        if '.' not in filename:
            filename += '.html'

        # Сохраняем HTML в файл
        filepath = os.path.join(HTML_DIRECTORY, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"HTML сохранен в {filepath}")
        return filename, response.text

    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        # Генерируем имя файла даже при ошибке
        parsed_url = urlparse(url)
        path = parsed_url.path.rstrip('/')
        filename = os.path.basename(path) if path else parsed_url.netloc.replace('.', '_')
        if '.' not in filename:
            filename += '.html'
        return filename, None

# %% [markdown]
# ## 4. Определение информативности страницы
#
# Не все страницы содержат полезный контент. Нужно отфильтровать:
# - Навигационные страницы (много ссылок, мало текста)
# - Страницы авторизации
# - Пустые страницы

# %%
def is_content_page(html_content: str) -> bool:
    """
    Определяет, является ли страница информативной.

    Args:
        html_content: HTML-содержимое страницы

    Returns:
        True, если страница содержит полезную информацию
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Удаляем скрипты и стили
        for tag in ['script', 'style']:
            for element in soup.find_all(tag):
                element.decompose()

        text = soup.get_text(separator=' ', strip=True)

        # Аналитика страницы
        word_count = len(text.split())
        link_count = len(soup.find_all('a', href=True))
        link_to_text_ratio = link_count / word_count if word_count > 0 else float('inf')

        # Критерии информативной страницы
        is_informative = (
            word_count > 300 or
            (word_count > 100 and link_count < 15) or
            (word_count > 150 and link_to_text_ratio < 0.1)
        )

        status = "Информативная" if is_informative else "Не информативная"
        print(f"  {status}: {word_count} слов, {link_count} ссылок")

        return is_informative

    except Exception as e:
        print(f"Ошибка при анализе страницы: {e}")
        return False

# %% [markdown]
# ## 5. Извлечение основного контента страницы
#
# Используем BeautifulSoup для поиска основного контента по популярным селекторам:
# `main`, `article`, `.content`, `#content` и т.д.

# %%
def extract_main_content(html_content: str, url: str) -> str:
    """
    Извлекает основной контент страницы и форматирует его.

    Args:
        html_content: HTML-содержимое страницы
        url: URL страницы

    Returns:
        Отформатированный HTML с основным контентом
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Удаляем ненужные элементы
        for tag in ['script', 'style', 'iframe', 'noscript']:
            for element in soup.find_all(tag):
                element.decompose()

        # Ищем основной контент по популярным селекторам
        content_selectors = [
            'main', 'article', '.content', '#content', '.main-content',
            '.page-content', '.container', '.article-content'
        ]

        main_content = None
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 200:
                main_content = content
                print(f"  Найден контент по селектору: {selector}")
                break

        if not main_content:
            main_content = soup.body
            print("  Используем body")

        page_title = soup.title.string if soup.title else 'Документ'

        # Создаём отформатированный HTML
        formatted_html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>{page_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{ color: #0066cc; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        h2, h3, h4 {{ color: #0066cc; }}
        a {{ color: #0066cc; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .footer {{ margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>{page_title}</h1>
    {main_content}
    <div class="footer">
        Источник: <a href="{url}">{url}</a><br>
        Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>"""

        return formatted_html

    except Exception as e:
        print(f"Ошибка при извлечении контента: {e}")
        return html_content

# %% [markdown]
# ## 6. Извлечение текста из HTML
#
# Структурированное извлечение текста с сохранением иерархии заголовков.

# %%
def extract_text_content(html_content: str) -> str:
    """
    Извлекает и форматирует текст из HTML.

    Args:
        html_content: HTML-содержимое страницы

    Returns:
        Извлечённый текст в Markdown-подобном формате
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Удаляем ненужные элементы
        for tag in ['script', 'style', 'iframe', 'noscript']:
            for element in soup.find_all(tag):
                element.decompose()

        title = soup.title.string if soup.title else ''
        formatted_text = f"{title}\n{'=' * len(title)}\n\n" if title else ""

        # Структурированное извлечение
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            tag_text = tag.get_text(strip=True)
            if tag_text:
                if tag.name.startswith('h'):
                    level = int(tag.name[1])
                    formatted_text += f"\n{'#' * level} {tag_text}\n"
                elif tag.name == 'p':
                    formatted_text += f"{tag_text}\n\n"
                elif tag.name == 'li':
                    formatted_text += f"- {tag_text}\n"

        # Fallback на обычное извлечение
        if len(formatted_text) < 200:
            formatted_text = (title + "\n\n") if title else ""
            formatted_text += soup.get_text(separator='\n', strip=True)

        # Очистка
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
        formatted_text = re.sub(r'\s{2,}', ' ', formatted_text)

        formatted_text += f"\n\n----------\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return formatted_text

    except Exception as e:
        print(f"Ошибка при извлечении текста: {e}")
        return ""

# %% [markdown]
# ## 7. Конвертация HTML в PDF с помощью Playwright
#
# Playwright позволяет рендерить HTML как в браузере и сохранять в PDF.

# %%
def generate_pdf_with_playwright(html_file: str, output_pdf: str) -> bool:
    """
    Конвертирует HTML-файл в PDF с помощью Playwright.

    Args:
        html_file: Путь к HTML-файлу
        output_pdf: Путь для сохранения PDF

    Returns:
        True при успехе, False при ошибке
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f'file:{html_file}')
            page.pdf(path=output_pdf)
            browser.close()
        return True
    except Exception as e:
        print(f"Ошибка Playwright: {e}")
        return False


def convert_html_to_pdf(html_content: str, output_file: str) -> bool:
    """
    Конвертирует HTML в PDF.

    Args:
        html_content: HTML-содержимое
        output_file: Путь для сохранения PDF

    Returns:
        True при успехе, False при ошибке
    """
    try:
        print(f"  Конвертация в PDF: {output_file}")

        # Сохраняем во временный файл
        temp_html_path = os.path.abspath('temp_convert.html')
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Конвертируем
        success = generate_pdf_with_playwright(temp_html_path, output_file)

        # Удаляем временный файл
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)

        if success and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"  PDF создан: {output_file}")
            return True
        else:
            print("  Ошибка: PDF не создан")
            return False

    except Exception as e:
        print(f"Ошибка при конвертации в PDF: {e}")
        return False

# %% [markdown]
# ## 8. Обработка одной страницы
#
# Объединяем все шаги: загрузка → проверка → извлечение → конвертация.

# %%
def process_webpage(url: str) -> bool:
    """
    Полная обработка одной веб-страницы.

    Args:
        url: URL страницы

    Returns:
        True при успешной обработке
    """
    print(f"\n=== Обработка: {url} ===")

    # Шаг 1: Загрузка
    filename, html_content = download_webpage(url)
    if not html_content:
        print("Не удалось загрузить страницу")
        return False

    # Шаг 2: Проверка информативности
    if not is_content_page(html_content):
        print("Страница не информативна, пропускаем")
        return False

    # Шаг 3: Извлечение контента
    formatted_html = extract_main_content(html_content, url)

    # Сохраняем очищенный HTML
    clean_html_path = os.path.join(HTML_DIRECTORY, f"clean_{filename}")
    with open(clean_html_path, 'w', encoding='utf-8') as f:
        f.write(formatted_html)
    print(f"  HTML: {clean_html_path}")

    # Шаг 4: Извлечение текста
    extracted_text = extract_text_content(html_content)
    text_filename = os.path.splitext(filename)[0] + '.txt'
    text_path = os.path.join(TEXT_DIRECTORY, text_filename)
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"  Текст: {text_path}")

    # Шаг 5: Конвертация в PDF
    pdf_filename = os.path.splitext(filename)[0] + '.pdf'
    pdf_path = os.path.join(PDF_DIRECTORY, pdf_filename)
    success = convert_html_to_pdf(formatted_html, pdf_path)

    return success

# %% [markdown]
# ## 9. Сортировка URL по глубине
#
# Более глубокие страницы обычно содержат больше контента.

# %%
def sort_urls_by_depth(urls: list[str]) -> list[str]:
    """Сортирует URL по глубине (количеству сегментов пути)."""
    def get_path_depth(url):
        parsed = urlparse(url)
        segments = [s for s in parsed.path.split('/') if s]
        return len(segments)

    return sorted(urls, key=get_path_depth, reverse=True)

# %% [markdown]
# ## 10. Пакетная обработка URL из sitemap

# %%
def process_urls_from_sitemap(website_urls: list[str], max_pages: int = 10) -> int:
    """
    Обрабатывает URL из карты сайта до достижения лимита информативных страниц.

    Args:
        website_urls: Список URL
        max_pages: Максимум страниц для обработки

    Returns:
        Количество успешно обработанных страниц
    """
    print(f"\n=== Начало обработки (лимит: {max_pages} страниц) ===")

    processed_count = 0
    content_pages = 0
    success_count = 0

    sorted_urls = sort_urls_by_depth(website_urls)
    urls_to_check = sorted_urls[:50]

    for i, url in enumerate(urls_to_check):
        print(f"\n[{i+1}/{len(urls_to_check)}] {url}")
        processed_count += 1

        filename, html_content = download_webpage(url)

        if not html_content:
            print("  Не удалось загрузить")
            continue

        if is_content_page(html_content):
            content_pages += 1
            print(f"  Информативная страница #{content_pages}")

            if process_webpage(url):
                success_count += 1

            if content_pages >= max_pages:
                print(f"\nДостигнут лимит ({max_pages})")
                break
        else:
            print("  Пропускаем")

        time.sleep(1)

    print("\n=== Итоги ===")
    print(f"Проверено URL: {processed_count}")
    print(f"Информативных: {content_pages}")
    print(f"Успешно обработано: {success_count}")

    return success_count

# %% [markdown]
# ## Запуск обработки
#
# Укажите URL sitemap вашего сайта и количество страниц для обработки.

# %%
# Пример запуска
if __name__ == "__main__":
    # Получаем список URL
    sitemap_url = 'https://www.tbank.ru/business/help/sitemap.xml'
    all_urls = get_urls_from_sitemap(sitemap_url)

    # Обрабатываем страницы
    num_processed = process_urls_from_sitemap(all_urls, max_pages=5)
    print(f"\nИтог: обработано {num_processed} страниц")

# %% [markdown]
# ## Выводы
#
# После выполнения лабораторной работы ответьте на вопросы:
#
# 1. Какие критерии вы использовали для определения информативности страницы?
# 2. С какими проблемами кодировки вы столкнулись?
# 3. Какие селекторы оказались наиболее эффективными для извлечения контента?
# 4. Как можно улучшить качество извлечённого текста?
#
# ---
#
# ## Ваши наблюдения
#
# *Запишите здесь свои выводы*

# %%
# Место для дополнительных экспериментов
