# %% [markdown]
# # 🔍 Лабораторная работа №7: Поисковая система с ReAct агентом в LangGraph
# 
# ## 📚 Введение
# 
# Добро пожаловать в лабораторную работу, посвященную созданию интеллектуальной поисковой системы! 
# В этой работе мы объединим векторное хранилище LanceDB с технологией ReAct (Reasoning and Action) 
# в фреймворке LangGraph для создания агента, способного понимать запросы, рассуждать и находить 
# релевантную информацию в документах.
# 
# **Почему это важно?** Простой текстовый поиск не всегда способен понять контекст и намерения пользователя. 
# Интеллектуальные агенты на основе ReAct могут:
# * Анализировать запрос и выбирать оптимальную стратегию поиска
# * Комбинировать рассуждения с действиями для получения лучших результатов
# * Давать обоснованные ответы, опираясь на найденную информацию
# 
# В этой лабораторной работе мы будем использовать:
# * **LanceDB** - эффективное векторное хранилище для поиска документов
# * **LangGraph** - фреймворк для создания структурированных графов языковых моделей
# * **ReAct** - подход, сочетающий рассуждения (reasoning) и действия (action)
# 
# ## 🎯 Чему вы научитесь:
# 
# 1. **🧩 Работа с векторными базами данных** - как подключаться и выполнять поиск
# 2. **🔄 Создание ReAct агента** - реализация паттерна рассуждения и действия
# 3. **📊 Использование LangGraph** - построение графа состояний для управления агентом
# 4. **🚀 Инструменты для поиска** - разработка специализированных инструментов 
# 5. **💾 Сохранение состояния** - реализация персистентности между сессиями
# 
# Готовы создать интеллектуального поискового агента? Давайте начнем!

# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
# Основные библиотеки
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Импорт компонентов LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Импорт компонентов для работы с векторным хранилищем
from langchain_community.vectorstores import LanceDB

# Импорт компонентов LangGraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict, List, Dict, Any

# Инструменты для ReAct агента
from langchain_core.tools import Tool

# %% [markdown]
# ## 1. Подключение к векторной базе данных
# 
# **💡 Почему это важно**: Векторная база данных хранит документы в виде математических векторов, что позволяет
# искать документы не только по точному соответствию ключевых слов, но и по семантической близости.
# 
# Мы будем использовать LanceDB из лабораторной работы №4, где уже хранятся документы Т-Банка.
# 
# **🔍 В этом разделе мы**:
# * Подключимся к существующей базе данных LanceDB
# * Проверим содержимое базы данных
# * Подготовим векторное хранилище для поиска

# %%
def connect_to_lancedb(db_path="../lab4_pdf_pipeline/data/lancedb", table_name="pdf_docs"):
    """
    Подключение к существующей базе данных LanceDB.
    
    Args:
        db_path: Путь к директории базы данных LanceDB
        table_name: Название таблицы в базе данных
    
    Returns:
        Экземпляр LanceDB для работы с векторным хранилищем
    """
    print(f"Подключение к базе данных LanceDB по пути: {db_path}")

    try:
        # Пытаемся импортировать lancedb
        import lancedb
        
        # Подключение к базе данных
        db = lancedb.connect(db_path)
        
        # Проверка существования таблицы
        table_names = db.table_names()
        
        if not table_names:
            print(f"База данных не содержит таблиц. Путь: {db_path}")
            return None
        
        if table_name not in table_names:
            print(f"Таблица {table_name} не найдена. Доступные таблицы: {table_names}")
            return None
        
        # Открываем таблицу
        table = db.open_table(table_name)
        
        # Вывод информации о таблице
        print(f"Успешное подключение к таблице: {table_name}")
        print(f"Количество документов в базе: {table.count_rows()}")

        # Создаем модель эмбеддингов
        embeddings = MistralAIEmbeddings()
        
        # Создаем экземпляр LanceDB для LangChain
        vector_store = LanceDB(
            connection=db,
            table_name=table_name,
            embedding=embeddings
        )
        
        return vector_store
    except ImportError:
        print("Ошибка импорта lancedb. Убедитесь, что библиотека установлена.")
        return None
    except Exception as e:
        print(f"Ошибка при подключении к базе данных: {str(e)}")
        return None

# %%
# Проверка подключения к базе данных
vector_db = connect_to_lancedb()

# %% [markdown]
# ## 2. Создание инструментов для поиска
# 
# **💡 Ключевой концепт**: Инструменты (tools) в ReAct позволяют агенту выполнять действия, такие как поиск информации.
# Хорошо продуманные инструменты делают агента более эффективным и целенаправленным.
# 
# **🔍 В этом разделе мы**:
# * Разработаем инструмент для стандартного векторного поиска
# * Создадим инструмент для поиска с фильтрацией по метаданным
# * Добавим инструмент для анализа найденных документов

# %%
# 
def raw_search_documents(query, vector_store=None, k=3):
    """
    Чистая функция поиска документов без зависимостей от инструментов LangChain.
    Эта функция предотвращает конфликты между объектом vector_store и системой обратных вызовов.
    
    Args:
        query (str): Текстовый запрос для поиска
        vector_store: Векторное хранилище LanceDB для поиска (LanceDB)
        k (int): Количество документов для возврата
    
    Returns:
        str: Отформатированная строка с найденными документами и их метаданными
    """
    try:
        
        
        # Выполнение поиска через LangChain API - семантический поиск по векторам
        results = vector_store.similarity_search(query, k=k)
        
        # Форматирование результатов в читаемый вид
        output = f'По запросу "{query}" найдено {len(results)} документов:\n\n'
        
        for i, doc in enumerate(results):
            output += f"Документ {i+1}:\n"
            output += f"{doc.page_content}\n\n"
            if hasattr(doc, 'metadata') and doc.metadata:
                output += f"Метаданные: {doc.metadata}\n\n"
        
        return output
    except Exception as e:
        return f"Ошибка при выполнении поиска: {str(e)}"

def raw_search_with_filter(query, metadata_filter, vector_store=None, k=3):
    """
    Чистая функция поиска документов с применением фильтра по метаданным.
    Работает напрямую с векторным хранилищем без использования инструментов LangChain.
    
    Args:
        query (str): Текстовый запрос для поиска
        metadata_filter (dict): Словарь с фильтрами для метаданных
        vector_store: Векторное хранилище LanceDB для поиска
        k (int): Количество документов для возврата
    
    Returns:
        str: Отформатированная строка с найденными документами и их метаданными
    """
    try:
        results = vector_store.similarity_search(
            query=query, 
            k=k,
            filter=metadata_filter
        )
        
        # Форматирование результатов в читаемый вид
        output = f'По запросу "{query}" с фильтром {metadata_filter} найдено {len(results)} документов:\n\n'
        
        for i, doc in enumerate(results):
            output += f"Документ {i+1}:\n"
            output += f"{doc.page_content}\n\n"
            if hasattr(doc, 'metadata') and doc.metadata:
                output += f"Метаданные: {doc.metadata}\n\n"
        
        return output
    except Exception as e:
        return f"Произошла ошибка при поиске с фильтром: {str(e)}"

def raw_analyze_documents(documents):
    """
    Чистая функция анализа документов без связи с инструментами LangChain.
    Использует языковую модель для анализа содержимого документов.
    
    Args:
        documents (str): Строка с текстом документов для анализа
    
    Returns:
        str: Структурированный анализ документов с ключевой информацией
    """
    # Создаем шаблон запроса для анализа документов
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - эксперт по анализу документов. 
        Проанализируйте предоставленные документы и выделите:
        1. Ключевые темы и концепты
        2. Важные факты и цифры
        3. Основные выводы
        
        Представьте ваш анализ в структурированном формате."""),
        ("user", "{documents}")
    ])
    
    # Используем модель GPT-4o-mini для анализа документов
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    chain = prompt | model | StrOutputParser()
    
    return chain.invoke({"documents": documents})

# %% [markdown]
# ## 3. Определение состояния ReAct агента
# 
# **💡 Ключевой концепт**: Состояние агента в LangGraph хранит всю необходимую информацию для принятия решений
# и выполнения действий. Для ReAct агента нам необходимо хранить историю сообщений, результаты поиска и
# информацию о текущем статусе обработки запроса.
# 
# **🔍 В этом разделе мы**:
# * Определим структуру состояния для поискового агента
# * Создадим функцию для инициализации начального состояния

# %%
class SearchAgentState(TypedDict):
    """
    Состояние поискового агента на основе ReAct.
    """
    # Сообщения в диалоге (используем reducer add_messages для автоматического добавления)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Текущий статус обработки запроса
    status: str
    # Результаты последнего поиска
    search_results: List[str]
    # История поисковых запросов
    search_history: List[str]

def create_empty_state() -> SearchAgentState:
    """
    Создание начального пустого состояния для агента.
    """
    return {
        "messages": [],
        "status": "waiting_for_query",
        "search_results": [],
        "search_history": []
    }

# %% [markdown]
# ## 4. Создание узлов для графа ReAct
# 
# **💡 Ключевой концепт**: ReAct состоит из рассуждений (reasoning) и действий (actions). В LangGraph
# это реализуется через отдельные узлы, которые выполняют различные функции в процессе обработки запроса.
# 
# **🔍 В этом разделе мы**:
# * Создадим узел для анализа запроса и планирования поиска
# * Реализуем узел для выполнения поиска
# * Добавим узел для формирования ответа на основе найденной информации

# %%
def create_search_agent_nodes(vector_store):
    """
    Создание узлов для поискового агента на основе ReAct.
    
    Args:
        vector_store: Векторное хранилище для поиска
        
    Returns:
        Словарь с узлами графа
    """
    # Инициализация модели
    model = ChatMistralAI(model="mistral-small-latest", temperature=0.2)
    
    # Вспомогательные функции для работы с чистыми функциями поиска
    def _search_func(query_str):
        """Прямая реализация поиска без вызова инструмента"""
        return raw_search_documents(query_str, vector_store)
    
    def _filter_search_func(args_str):
        """Прямая реализация поиска с фильтром без вызова инструмента"""
        import json
        try:
            args = json.loads(args_str)
            query = args.get("query", "")
            metadata_filter = args.get("metadata_filter", {})
            return raw_search_with_filter(query, metadata_filter, vector_store)
        except Exception as e:
            return f"Ошибка при обработке аргументов для фильтрованного поиска: {str(e)}"
    
    def _analyze_func(docs):
        """Прямая реализация анализа документов без вызова инструмента"""
        return raw_analyze_documents(docs)
    
    # Определяем инструменты, используя наши функции напрямую
    search_tool = Tool(
        name="search_documents",
        description="Поиск документов по текстовому запросу",
        func=_search_func
    )
    
    filtered_search_tool = Tool(
        name="search_with_filter",
        description="Поиск документов с фильтрацией по метаданным",
        func=_filter_search_func
    )
    
    analyze_tool = Tool(
        name="analyze_documents",
        description="Анализ найденных документов и извлечение ключевой информации",
        func=_analyze_func
    )
    
    # Список инструментов для модели
    tools = [
        search_tool,
        filtered_search_tool,
        analyze_tool
    ]
    
    # Связывание модели с инструментами
    model_with_tools = model.bind_tools(tools)
    
    # Узел для выполнения поиска
    def execute_search(state: SearchAgentState) -> SearchAgentState:
        """Выполняет поиск на основе проанализированного запроса."""
        # Извлечение последнего запроса пользователя
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return state  # Нет запросов пользователя
            
        query = user_messages[-1].content
        
        # Системное сообщение для определения подхода к поиску
        system_message = SystemMessage(content="""
        Вы - поисковый ассистент, который должен выбрать наиболее подходящий инструмент для поиска.
        
        У вас есть следующие инструменты:
        1. search_documents - стандартный поиск по запросу
        2. search_with_filter - поиск с фильтрацией по метаданным (например, фильтр по категории, дате, автору)
        
        Выберите инструмент на основе запроса пользователя и выполните поиск.
        Если запрос содержит указание на фильтрацию или категорию, используйте search_with_filter.
        В противном случае используйте search_documents.
        """)
        
        # Создание сообщений для модели
        messages = [
            system_message,
            HumanMessage(content=f"Запрос пользователя: {query}. Какой инструмент поиска лучше использовать?")
        ]
        
        # Вызов модели с инструментами для выбора и выполнения поиска
        response = model_with_tools.invoke(messages)
        
        # Проверяем, были ли использованы инструменты
        search_results = []
        search_tool_used = False
        updated_messages = state["messages"] + [response]
        
        # Обработка вызовов инструментов
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                # Определяем вызванный инструмент
                tool_name = None
                tool_args = {}
                tool_call_id = None
                
                # Для dict (новый API)
                if isinstance(tool_call, dict):
                    if 'name' in tool_call:
                        tool_name = tool_call['name']
                    if 'arguments' in tool_call:
                        tool_args = tool_call['arguments']
                    if 'id' in tool_call:
                        tool_call_id = tool_call['id']
                # Для объекта (старый API)
                elif hasattr(tool_call, "name"):
                    tool_name = tool_call.name
                    if hasattr(tool_call, "args"):
                        tool_args = tool_call.args
                    if hasattr(tool_call, "id"):
                        tool_call_id = tool_call.id
                
                # Выполняем вызов инструмента
                if tool_name in ["search_documents", "search_with_filter"]:
                    search_tool_used = True
                    tool_result = None
                    
                    # Вызов инструмента search_documents
                    if tool_name == "search_documents":
                        search_query = query
                        if isinstance(tool_args, dict) and 'query' in tool_args:
                            search_query = tool_args['query']
                        elif isinstance(tool_args, str):
                            import json
                            try:
                                args_dict = json.loads(tool_args)
                                search_query = args_dict.get('query', query)
                            except:
                                search_query = query
                        
                        # Прямой вызов функции поиска
                        tool_result = _search_func(search_query)
                        
                    # Вызов инструмента search_with_filter
                    elif tool_name == "search_with_filter":
                        metadata_filter = {}
                        search_query = query
                        
                        if isinstance(tool_args, dict):
                            if 'query' in tool_args:
                                search_query = tool_args['query']
                            if 'metadata_filter' in tool_args:
                                metadata_filter = tool_args['metadata_filter']
                        elif isinstance(tool_args, str):
                            import json
                            try:
                                args_dict = json.loads(tool_args)
                                search_query = args_dict.get('query', query)
                                metadata_filter = args_dict.get('metadata_filter', {})
                            except:
                                pass
                        
                        # Создаем строку аргументов для filtered_search_tool
                        args_str = json.dumps({"query": search_query, "metadata_filter": metadata_filter})
                        tool_result = _filter_search_func(args_str)
                    
                    # Добавляем результат вызова инструмента
                    if tool_result:
                        search_results.append(tool_result)
                        # Добавляем сообщение от инструмента
                        tool_message = ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id
                        )
                        updated_messages.append(tool_message)
        
        # Если инструменты не были использованы, выполняем поиск вручную
        if not search_tool_used:
            # Прямой вызов функции поиска
            result = _search_func(query)
            search_results.append(result)
        
        # Обновление истории поисков
        updated_history = state["search_history"] + [query]
        
        # Обновление состояния
        return {
            "messages": updated_messages,
            "status": "search_executed",
            "search_results": search_results,
            "search_history": updated_history
        }
        
    # Импортируем узел анализа запроса и узел формирования ответа из предыдущего кода
    def analyze_query(state: SearchAgentState) -> SearchAgentState:
        """Анализирует запрос пользователя и определяет стратегию поиска."""
        # Системное сообщение
        system_message = SystemMessage(content="""
        Вы - поисковый ассистент, использующий ReAct (Reasoning and Action) подход.
        
        Ваша задача - понять запрос пользователя и выбрать правильную стратегию поиска:
        1. Определить, требуется ли простой поиск или поиск с фильтрацией.
        2. Выделить ключевые слова и фразы для поиска.
        3. Оценить, достаточно ли информации в запросе для поиска.
        
        Сформулируйте свои рассуждения и план поиска.
        """)
        
        # Объединение системного сообщения с историей сообщений
        messages = [system_message] + state["messages"]
        
        # Вызов модели для анализа
        response = model.invoke(messages)
        
        # Обновление состояния
        return {
            "messages": state["messages"] + [response],
            "status": "query_analyzed",
            "search_results": state["search_results"],
            "search_history": state["search_history"]
        }
    
    def generate_response(state: SearchAgentState) -> SearchAgentState:
        """Формирует ответ на основе результатов поиска."""
        # Если нет результатов поиска, возвращаем сообщение об этом
        if not state["search_results"]:
            response = AIMessage(content="Извините, я не смог найти релевантную информацию по вашему запросу. Пожалуйста, попробуйте сформулировать запрос иначе.")
            return {
                "messages": state["messages"] + [response],
                "status": "completed",
                "search_results": state["search_results"],
                "search_history": state["search_history"]
            }
        
        # Системное сообщение для формирования ответа
        system_message = SystemMessage(content="""
        Вы - поисковый ассистент. Используйте результаты поиска для формирования
        информативного и полезного ответа на запрос пользователя.
        
        Структурируйте свой ответ следующим образом:
        1. Краткое резюме найденной информации
        2. Детальный ответ на вопрос пользователя, опираясь на найденные документы
        3. Указание источников информации (номера документов)
        
        Основывайтесь только на предоставленных результатах поиска.
        Если информации недостаточно, честно укажите на это.
        """)
        
        # Создание контекста с результатами поиска
        search_context = "\n\n".join(state["search_results"])
        search_context_message = SystemMessage(content=f"Результаты поиска:\n{search_context}")
        
        # Получение последнего запроса пользователя
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        last_user_message = user_messages[-1] if user_messages else None
        
        # Если нет запроса, возвращаем состояние без изменений
        if not last_user_message:
            return state
        
        # Создание запроса для формирования ответа
        query_for_response = f"Запрос пользователя: {last_user_message.content}. Сформируйте ответ на основе результатов поиска."
        
        # Сообщения для модели
        messages = [
            system_message,
            search_context_message,
            HumanMessage(content=query_for_response)
        ]
        
        # Вызов модели для формирования ответа
        response = model.invoke(messages)
        
        # Обновление состояния
        return {
            "messages": state["messages"] + [response],
            "status": "completed",
            "search_results": state["search_results"],
            "search_history": state["search_history"]
        }
    
    # Возвращаем словарь с узлами
    return {
        "analyze_query": analyze_query,
        "execute_search": execute_search,
        "generate_response": generate_response
    }

# %% [markdown]
# ## 5. Создание графа ReAct агента
# 
# **💡 Ключевой концепт**: Граф в LangGraph определяет, как узлы связаны между собой и какие переходы 
# возможны между ними. Для ReAct агента мы создадим граф с условными переходами, который позволит
# выполнять разные действия в зависимости от статуса обработки запроса.
# 
# **🔍 В этом разделе мы**:
# * Создадим граф состояний для агента
# * Определим условные переходы между узлами
# * Скомпилируем граф для дальнейшего использования

# %%
def create_search_agent_graph(vector_store):
    """
    Создание графа ReAct агента для поиска.
    
    Args:
        vector_store: Векторное хранилище для поиска
        
    Returns:
        Скомпилированный граф
    """
    # Создание графа с определенным типом состояния
    workflow = StateGraph(SearchAgentState)
    
    # Получение узлов
    nodes = create_search_agent_nodes(vector_store)
    
    # Добавление узлов в граф
    for name, function in nodes.items():
        workflow.add_node(name, function)
    
    # Определение условных переходов
    def should_search(state: SearchAgentState) -> str:
        """Определяет, нужно ли выполнять поиск или переходить к генерации ответа."""
        if state["status"] == "query_analyzed":
            return "execute_search"
        else:
            return "generate_response"
    
    # Добавление ребер с условными переходами
    workflow.add_edge(START, "analyze_query")
    workflow.add_conditional_edges("analyze_query", should_search)
    workflow.add_edge("execute_search", "generate_response")
    
    # Компиляция графа
    graph = workflow.compile()
    
    return graph

# %% [markdown]
# ## 6. Демонстрация работы ReAct агента
# 
# **💡 Ключевой концепт**: Теперь, когда мы создали все необходимые компоненты, пора увидеть нашего 
# агента в действии. Мы проверим, как он анализирует запросы, выполняет поиск и формирует ответы.
# 
# **🔍 В этом разделе мы**:
# * Проверим работу агента на различных типах запросов
# * Отследим прохождение запроса через все этапы обработки
# * Оценим качество ответов и их релевантность

# %%
def demonstrate_search_agent(vector_store=None):
    """
    Демонстрация работы поискового агента на основе ReAct.
    Показывает полный цикл обработки запроса: анализ, поиск и формирование ответа.
    
    Args:
        vector_store: Векторное хранилище для поиска
    
    Returns:
        dict: Конечное состояние агента после обработки всех запросов
    """
    # Создание графа для ReAct агента
    graph = create_search_agent_graph(vector_store)
    
    # Начальное состояние
    state = create_empty_state()
    
    # Список запросов для демонстрации различных сценариев поиска
    test_queries = [
        "Какие тарифы предлагает Т-Банк?",
        "Расскажи о кредитных картах и условиях их получения",
        "Какие инвестиционные продукты доступны для клиентов?"
    ]
    
    # Выполнение запросов
    for i, query in enumerate(test_queries):
        print(f"\n{'='*50}")
        print(f"ДЕМОНСТРАЦИЯ {i+1}: {query}")
        print(f"{'='*50}\n")
        
        # Добавление запроса пользователя в состояние
        state["messages"].append(HumanMessage(content=query))
        
        # Вызов графа для обработки запроса
        state = graph.invoke(state)
        
        # Вывод ответа ассистента
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        last_message = ai_messages[-1] if ai_messages else None
        
        if last_message:
            print(f"📝 ОТВЕТ АГЕНТА:\n{last_message.content}")
        
        # Вывод информации о процессе обработки
        print(f"\n📊 СТАТИСТИКА:")
        print(f"• Статус: {state['status']}")
        print(f"• Найдено документов: {len(state['search_results'])}")
        print(f"• Выполнено поисков: {len(state['search_history'])}")
        print(f"{'='*50}\n")
    
    return state

# %% [markdown]
# ## 7. Сохранение состояния агента (персистентность)
# 
# **💡 Ключевой концепт**: Для создания полноценной поисковой системы нам нужно обеспечить сохранение
# состояния между сессиями пользователя. LangGraph поддерживает checkpointing, который позволяет
# сохранять и восстанавливать состояние агента.
# 
# **🔍 В этом разделе мы**:
# * Добавим механизм сохранения состояния с использованием MemorySaver
# * Продемонстрируем, как состояние можно восстановить между сессиями
# * Реализуем многопользовательский режим с использованием thread_id

# %%
def demonstrate_persistence():
    """
    Демонстрация персистентности с использованием чекпоинтеров в LangGraph.
    Показывает сохранение состояния между сессиями.
    """
    print("\n=== Демонстрация персистентности с чекпоинтерами ===\n")
    
    # Импортируем чекпоинтер для сохранения в памяти
    from langgraph.checkpoint.memory import MemorySaver
    
    # Определим простой процессор сообщений для демонстрации
    def simple_processor(state: SearchAgentState) -> SearchAgentState:
        """Простой процессор для демонстрации работы с состоянием."""
        # Получаем последнее сообщение пользователя
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return state
            
        last_message = user_messages[-1].content
        
        # Добавляем ответ ассистента
        response = AIMessage(content=f"Я получил ваше сообщение: '{last_message}'. Оно сохранено в истории.")
        
        # Обновляем историю поисковых запросов
        updated_history = state["search_history"] + [last_message]
        
        # Возвращаем обновленное состояние
        return {
            "messages": state["messages"] + [response],
            "status": "completed",
            "search_results": state["search_results"],
            "search_history": updated_history
        }
    
    # Создание графа с определенным типом состояния
    workflow = StateGraph(SearchAgentState)
    
    # Добавляем узел в граф
    workflow.add_node("process_message", simple_processor)
    
    # Добавляем ребро от начала к узлу обработки
    workflow.add_edge(START, "process_message")
    
    # Создаем чекпоинтер в памяти
    memory_saver = MemorySaver()
    
    # Компилируем граф с чекпоинтером
    graph = workflow.compile(checkpointer=memory_saver)
    
    print("1️⃣ Создали граф с чекпоинтером MemorySaver")
    
    # Создаем начальное состояние
    state = create_empty_state()
    
    # Создаем уникальный идентификатор для сессии
    session_id = "demo_session_1"
    config = {"configurable": {"thread_id": session_id}}
    
    print("\n2️⃣ Первая сессия - отправляем сообщение")
    
    # Добавляем сообщение пользователя
    message1 = "Это мое первое сообщение, которое должно сохраниться в истории"
    state["messages"].append(HumanMessage(content=message1))
    
    # Вызываем граф с конфигурацией для сохранения состояния
    print(f"👤 Пользователь: {message1}")
    state = graph.invoke(state, config=config)
    
    # Выводим ответ
    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    last_message = ai_messages[-1] if ai_messages else None
    
    if last_message:
        print(f"🤖 Ассистент: {last_message.content}")
    
    print(f"📊 История запросов: {state['search_history']}")
    
    print("\n3️⃣ Проверяем сохранение состояния")
    
    # Получаем сохраненное состояние из чекпоинтера
    state_snapshot = graph.get_state(config)
    
    if state_snapshot:
        print("✅ Состояние успешно сохранено!")
        print(f"📝 Количество сообщений в истории: {len(state_snapshot.values['messages'])}")
        print(f"🔍 История запросов: {state_snapshot.values['search_history']}")
    
    print("\n4️⃣ Имитируем новую сессию - восстанавливаем состояние и добавляем новое сообщение")
    
    # Получаем сохраненное состояние для новой сессии
    restored_state = state_snapshot.values if state_snapshot else create_empty_state()
    
    # Добавляем новое сообщение в восстановленное состояние
    message2 = "Это мое второе сообщение. Помнишь ли ты первое?"
    restored_state["messages"].append(HumanMessage(content=message2))
    
    # Вызываем граф с тем же thread_id
    print(f"👤 Пользователь: {message2}")
    updated_state = graph.invoke(restored_state, config=config)
    
    # Выводим ответ
    ai_messages = [msg for msg in updated_state["messages"] if isinstance(msg, AIMessage)]
    last_message = ai_messages[-1] if ai_messages else None
    
    if last_message:
        print(f"🤖 Ассистент: {last_message.content}")
    
    print(f"📊 История запросов после восстановления: {updated_state['search_history']}")
    
    print("\n5️⃣ Итоговая проверка состояния")
    
    # Получаем финальное состояние из чекпоинтера
    final_snapshot = graph.get_state(config)
    
    if final_snapshot:
        print("✅ Итоговое состояние успешно сохранено!")
        print(f"📝 Количество сообщений в истории: {len(final_snapshot.values['messages'])}")
        
        # Выводим всю историю сообщений
        print("\n📜 Полная история диалога:")
        for i, msg in enumerate(final_snapshot.values["messages"]):
            if isinstance(msg, HumanMessage):
                print(f"   👤 Пользователь ({i+1}): {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"   🤖 Ассистент ({i+1}): {msg.content}")
    
    print("\n=== Демонстрация персистентности завершена ===")
    
    return memory_saver

# %% [markdown]
# ## 8. Поддержка множества пользователей
# 
# **💡 Ключевой концепт**: Для полноценной поисковой системы нужна поддержка нескольких пользователей
# одновременно. LangGraph позволяет это делать через использование разных thread_id для каждого пользователя.
# 
# **🔍 В этом разделе мы**:
# * Продемонстрируем, как обслуживать нескольких пользователей одновременно
# * Реализуем изоляцию состояний между пользователями

# %%
def demonstrate_multi_user_support(vector_store=None):
    """
    Демонстрация поддержки нескольких пользователей с изолированными состояниями.
    
    Args:
        vector_store: Векторное хранилище для поиска
    """
    from langgraph.checkpoint.memory import MemorySaver
    
    # Создание графа
    workflow = StateGraph(SearchAgentState)
    
    # Получение узлов
    nodes = create_search_agent_nodes(vector_store)
    
    # Добавление узлов
    for name, function in nodes.items():
        workflow.add_node(name, function)
    
    # Определение условных переходов
    def should_search(state: SearchAgentState) -> str:
        if state["status"] == "query_analyzed":
            return "execute_search"
        else:
            return "generate_response"
    
    # Добавление ребер
    workflow.add_edge(START, "analyze_query")
    workflow.add_conditional_edges("analyze_query", should_search)
    workflow.add_edge("execute_search", "generate_response")
    
    # Создание чекпоинтера
    memory_saver = MemorySaver()
    
    # Компиляция графа с чекпоинтером
    graph = workflow.compile(checkpointer=memory_saver)
    
    # Создание пользователей
    user_ids = ["user_1", "user_2"]
    user_states = {}
    
    # Инициализация состояний для пользователей
    for user_id in user_ids:
        user_states[user_id] = create_empty_state()
    
    # Запросы для пользователей
    user_queries = {
        "user_1": [
            "Какие депозиты предлагает Т-Банк?",
            "А какие есть варианты для бизнеса?"
        ],
        "user_2": [
            "Можно ли получить кредит онлайн?",
            "Какие документы нужны для ипотеки?"
        ]
    }
    
    # Обработка запросов каждого пользователя
    for user_id in user_ids:
        print(f"\n=== Пользователь {user_id} ===\n")
        
        config = {"configurable": {"thread_id": user_id}}
        state = user_states[user_id]
        
        for query in user_queries[user_id]:
            # Добавление запроса в состояние
            state["messages"].append(HumanMessage(content=query))
            
            # Вызов графа с соответствующим thread_id
            state = graph.invoke(state, config=config)
            
            # Вывод запроса и ответа
            print(f"Запрос: {query}")
            
            ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
            last_message = ai_messages[-1] if ai_messages else None
            
            if last_message:
                print(f"Ответ: {last_message.content}\n")
            
            # Обновление состояния пользователя
            user_states[user_id] = state
    
    # Проверка изоляции состояний
    print("\n=== Проверка изоляции состояний ===\n")
    
    for user_id in user_ids:
        # Получаем сохраненное состояние
        config = {"configurable": {"thread_id": user_id}}
        state_snapshot = graph.get_state(config)
        
        if state_snapshot:
            print(f"Пользователь {user_id}:")
            print(f"Количество сообщений: {len(state_snapshot.values['messages'])}")
            print(f"История поисковых запросов: {state_snapshot.values['search_history']}")
            print()
    
    return memory_saver

# %% [markdown]
# ## Заключение
# 
# **🎓 Что вы узнали в этой лаборатории**:
# 
# 1. **🧩 Интеграция с векторными базами данных** - как подключиться к LanceDB и выполнять поиск по документам
# 2. **🔄 Принципы паттерна ReAct** - как комбинировать рассуждения и действия для более эффективного поиска
# 3. **🚀 Создание агентов в LangGraph** - как определять состояние, узлы и переходы в графе
# 4. **💾 Сохранение состояния** - как реализовать персистентность и поддержку нескольких пользователей
# 

# %%
# Запуск демонстрации
if __name__ == "__main__":
    import lancedb
    import os
    
    # Путь к базе данных
    db_path = "../lab4_pdf_pipeline/data/lancedb"
    
    # Проверяем, существует ли путь к БД
    if not os.path.exists(db_path):
        print(f"Путь к базе данных не существует: {db_path}")
        print("Используем тестовое окружение...")
        
        # Используем тестовое окружение
        vector_db = None
    else:
        try:
            # Подключаемся к базе данных
            vector_db = connect_to_lancedb(db_path=db_path, table_name="pdf_docs")
            print("Успешное подключение к базе данных")
        except Exception as e:
            print(f"Ошибка при подключении к базе данных: {str(e)}")
            print("Используем тестовое окружение...")
            vector_db = None
    
    # Запускаем демонстрации
    print("\n=== Начинаем демонстрацию поискового агента ===\n")
    
    try:
        # Демонстрация поискового агента
        demonstrate_search_agent(vector_db)
    except Exception as e:
        print(f"Ошибка при демонстрации поискового агента: {str(e)}")
    
    try:
        # Демонстрация персистентности
        memory_saver = demonstrate_persistence()
        print("\nДемонстрация персистентности выполнена успешно.")
    except Exception as e:
        print(f"Ошибка при демонстрации персистентности: {str(e)}")

    try:
        # Демонстрация поискового агента
        demonstrate_multi_user_support(vector_db)
    except Exception as e:
        print(f"Ошибка при демонстрации поискового агента: {str(e)}")
    
    print("\n=== Демонстрации завершены ===\n")

# Узел для использования инструментов моделью
def create_tool_node(model_with_tools):
    """Создает узел графа для выполнения действий с помощью инструментов."""
    def tool_node(state: SearchAgentState) -> SearchAgentState:
        """Узел для работы с инструментами (tools)."""
        # Системное сообщение для определения использования инструментов
        system_message = SystemMessage(content="""
        Вы - поисковый ассистент, который использует инструменты для поиска информации. 
        
        Исходя из предыдущего контекста и запроса пользователя, используйте инструменты:
        1. search_documents - для поиска документов по запросу
        2. search_with_filter - для поиска с применением фильтров
        3. analyze_documents - для анализа найденных документов
        
        Вызывайте инструменты для получения нужной информации.
        """)
        
        # Подготовка сообщений (включая историю)
        messages = [system_message] + state["messages"]
        
        # Вызов модели с инструментами
        response = model_with_tools.invoke(messages)
        
        # Извлечение результатов использования инструментов
        search_results = []
        
        # Проверяем вызовы инструментов в ответе
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                # Проверяем, является ли tool_call словарем (новый API) или объектом (старый API)
                if isinstance(tool_call, dict):
                    # Новый API - tool_call это словарь
                    if 'output' in tool_call:
                        search_results.append(tool_call['output'])
                else:
                    # Старый API - tool_call это объект с атрибутами
                    if hasattr(tool_call, "output"):
                        search_results.append(tool_call.output)
        
        # Обновление состояния
        return {
            "messages": state["messages"] + [response],
            "status": "tool_used" if search_results else "no_tool_used",
            "search_results": search_results if search_results else state["search_results"],
            "search_history": state["search_history"]
        }
    
    return tool_node
