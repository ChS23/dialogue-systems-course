# %% [markdown]
# # 🧠 Лабораторная работа: Память и управление состоянием в диалоговых системах
# 
# ## 📚 Введение
# 
# Добро пожаловать в лабораторную работу, посвященную системам памяти в диалоговых приложениях! 
# В этой работе мы изучим, как создавать диалоговые системы, способные запоминать контекст разговора и 
# использовать его для более осмысленных ответов.
# 
# **Почему это важно?** Представьте диалоговую систему без памяти - она будет как человек с амнезией, 
# который не помнит ничего из предыдущего разговора. Чтобы создать по-настоящему полезного AI-ассистента, 
# нам необходимо:
# * Хранить историю взаимодействия с пользователем
# * Управлять состоянием диалога
# * Организовывать долгосрочную и краткосрочную память
# * Строить сложные потоки взаимодействия с пользователем
# 
# В этой лабораторной работе мы будем использовать два дополняющих друг друга фреймворка: 
# **LangChain** для работы с различными типами памяти и **LangGraph** для управления состоянием 
# и построения сложных диалоговых графов.
# 
# ## 🎯 Чему вы научитесь:
# 
# 1. **🧩 Базовые принципы памяти** - как сохранять и использовать контекст разговора
# 2. **🔄 Типы памяти в LangChain** - различные реализации систем памяти и их применение
# 3. **🚀 Управление состоянием** - как хранить и обновлять состояние диалоговой системы
# 4. **📊 Графы диалога** - создание сложных диалоговых потоков с помощью LangGraph
# 5. **🔍 Персистентность** - сохранение памяти между сессиями
# 6. **🛠️ Практические примеры** - разработка диалоговых систем с памятью и состоянием
# 
# Готовы создать по-настоящему умного собеседника, который помнит ваши предпочтения и контекст разговора? Давайте начнем!
# 

# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
# Основные библиотеки

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()
import os

# Импорт компонентов LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_mistralai import ChatMistralAI

# Компоненты для работы с памятью
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory, ConversationTokenBufferMemory

# Импорт компонентов LangGraph
from langgraph.graph import StateGraph, START, MessagesState

# %% [markdown]
# ## 1. Основы памяти в диалоговых системах
# 
# **💡 Почему это важно**: Без памяти о предыдущих взаимодействиях диалоговая система не сможет поддерживать осмысленный разговор. Пользователю придётся повторять контекст при каждом запросе, что делает общение неестественным и утомительным.
# 
# В самом простом виде, память - это список предыдущих сообщений, которые передаются языковой модели в качестве контекста. Давайте начнем с самого базового подхода - прямой передачи истории сообщений.
# 
# **🔍 В этом разделе мы**:
# * Создадим простую диалоговую систему с использованием `MessagesPlaceholder`
# * Покажем, как сохранить и передать историю сообщений
# * Проверим, как система использует контекст для формирования ответов

# %%
def demonstrate_basic_memory():
    """
    Демонстрация самого простого способа сохранения и использования истории диалога
    через прямую передачу сообщений в промпт.
    """
    # Создаем промпт, включающий плейсхолдер для сообщений
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - дружелюбный и полезный ассистент. 
        Отвечайте на вопросы пользователя, опираясь на контекст предыдущего разговора.
        Если в контексте есть противоречия, вежливо укажите на них.
        """),
        # MessagesPlaceholder позволяет вставить список сообщений в промпт
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Инициализируем модель
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    
    # Создаем цепочку: промпт -> модель -> парсер
    chain = prompt | model | StrOutputParser()
    
    # Создаем пустой список для хранения истории сообщений
    conversation_history = []
    
    # Первое взаимодействие
    user_input1 = "Привет! Меня зовут Алексей, и я интересуюсь искусственным интеллектом."
    
    # Добавляем сообщение пользователя в историю
    conversation_history.append(HumanMessage(content=user_input1))
    
    # Вызываем модель
    response1 = chain.invoke({
        "history": conversation_history,
        "input": user_input1
    })
    
    # Выводим ответ
    print(f"\nПользователь: {user_input1}")
    print(f"Ассистент: {response1}")
    
    # Добавляем ответ модели в историю
    conversation_history.append(AIMessage(content=response1))
    
    # Второе взаимодействие - проверяем, помнит ли система имя пользователя
    user_input2 = "Расскажи мне о последних достижениях в области компьютерного зрения."
    
    # Вызываем модель снова, передавая обновленную историю
    response2 = chain.invoke({
        "history": conversation_history,
        "input": user_input2
    })
    
    # Выводим ответ
    print(f"\nПользователь: {user_input2}")
    print(f"Ассистент: {response2}")
    
    # Добавляем новые сообщения в историю
    conversation_history.append(HumanMessage(content=user_input2))
    conversation_history.append(AIMessage(content=response2))
    
    # Третье взаимодействие - проверяем, помнит ли система предыдущий вопрос
    user_input3 = "А какие еще области ИИ сейчас активно развиваются помимо компьютерного зрения?"
    
    # Вызываем модель в третий раз
    response3 = chain.invoke({
        "history": conversation_history,
        "input": user_input3
    })
    
    # Выводим ответ
    print(f"\nПользователь: {user_input3}")
    print(f"Ассистент: {response3}")
    
    return conversation_history

# Запускаем функцию
print("\n=== 1. Основы памяти в диалоговых системах ===")
basic_memory_conversation = demonstrate_basic_memory()

# %% [markdown]
# В примере выше мы увидели, как можно передавать историю разговора через список сообщений. Этот подход работает, но имеет несколько недостатков:
# 
# 1. **Ручное управление**: Нам приходится вручную добавлять каждое сообщение в историю
# 2. **Отсутствие оптимизации**: История может становиться очень длинной и превышать контекстное окно модели
# 3. **Отсутствие персистентности**: История существует только в памяти программы
# 
# В следующих разделах мы рассмотрим более продвинутые подходы к работе с памятью, которые решают эти проблемы.

# %% [markdown]
# ## 2. Системы памяти в LangChain
# 
# **💡 Ключевой концепт**: LangChain предоставляет различные типы памяти, которые автоматизируют сохранение и обработку истории диалога.
# 
# В предыдущем примере мы вручную управляли списком сообщений. LangChain упрощает эту задачу с помощью классов памяти, которые:
# * Автоматически сохраняют историю сообщений
# * Предоставляют различные стратегии управления размером истории
# * Могут интегрироваться с внешними хранилищами для долгосрочного хранения
# 
# **🔍 Рассмотрим следующие типы памяти**:
# * `ChatMessageHistory` - базовый класс для хранения сообщений
# * `ConversationBufferMemory` - сохраняет всю историю разговора
# * `ConversationBufferWindowMemory` - сохраняет только N последних сообщений
# * `ConversationTokenBufferMemory` - ограничивает историю по количеству токенов
# * `ConversationSummaryMemory` - создает и обновляет сводку разговора

# %%
def demonstrate_chat_message_history():
    """
    Демонстрация использования ChatMessageHistory для хранения истории сообщений.
    """
    # Создаем экземпляр ChatMessageHistory для хранения сообщений
    message_history = ChatMessageHistory()
    
    # Добавляем сообщения
    message_history.add_user_message("Привет! Меня зовут Мария. Я хочу узнать больше о нейронных сетях.")
    message_history.add_ai_message("Здравствуйте, Мария! Рад помочь вам узнать о нейронных сетях. С чего бы вы хотели начать?")
    message_history.add_user_message("Расскажи мне о типах нейронных сетей и их применении.")
    
    # Выводим все сообщения в истории
    print("\nИстория сообщений через ChatMessageHistory:")
    for message in message_history.messages:
        if isinstance(message, HumanMessage):
            print(f"Пользователь: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"Ассистент: {message.content}")
    
    # Создаем модель
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    
    # Создаем промпт с плейсхолдером для истории сообщений
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы - эксперт по нейронным сетям. Отвечайте на вопросы подробно и понятно."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Создаем цепочку
    chain = prompt | model | StrOutputParser()
    
    # Используем сохраненные сообщения для запроса к модели
    result = chain.invoke({"messages": message_history.messages})
    
    print("\nОтвет модели при использовании ChatMessageHistory:")
    print(f"Ассистент: {result}")
    
    # Добавляем новый ответ в историю
    message_history.add_ai_message(result)
    
    return message_history

# Запускаем функцию
print("\n=== 2.1 Использование ChatMessageHistory ===")
message_history_example = demonstrate_chat_message_history()

# %%
def demonstrate_memory_types():
    """
    Демонстрация различных типов памяти в LangChain и их особенностей.
    """
    # Определяем общий промпт для всех экспериментов
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы - дружелюбный помощник. Отвечайте кратко и информативно."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Инициализируем модель
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    
    # Создаем цепочку: промпт -> модель -> парсер
    chain = prompt | model | StrOutputParser()
    
    # 1. ConversationBufferMemory - хранит все сообщения
    buffer_memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    
    # 2. ConversationBufferWindowMemory - хранит только k последних сообщений
    window_memory = ConversationBufferWindowMemory(k=2, return_messages=True, memory_key="history")
    
    # 3. ConversationTokenBufferMemory - ограничивает историю по количеству токенов
    token_memory = ConversationTokenBufferMemory(
        llm=model, max_token_limit=100, return_messages=True, memory_key="history"
    )
    
    # 4. ConversationSummaryMemory - создает и обновляет сводку разговора
    summary_memory = ConversationSummaryMemory(
        llm=model, return_messages=True, memory_key="history"
    )
    
    # Список всех типов памяти для тестирования
    memories = {
        "Buffer Memory": buffer_memory,
        "Window Memory (k=2)": window_memory,
        "Token Buffer Memory": token_memory,
        "Summary Memory": summary_memory
    }
    
    # Последовательность сообщений для теста
    test_messages = [
        "Привет! Меня зовут Иван.",
        "Какая сегодня погода в Москве?",
        "Спасибо. А что насчет прогноза на завтра?",
        "Понятно. Расскажи о популярных местах в Москве.",
        "А где можно вкусно поесть в центре?",
        "Теперь вернемся к погоде. Какая погода будет на выходных?"
    ]
    
    # Результаты для каждого типа памяти
    results = {}
    
    # Тестируем каждый тип памяти
    for memory_name, memory in memories.items():
        print(f"\n--- Тестирование {memory_name} ---")
        
        # Последовательно обрабатываем сообщения
        for i, message in enumerate(test_messages):
            # Сохраняем пару ввод-вывод в памяти
            memory.save_context({"input": message}, {"output": f"Ответ на сообщение {i+1}"})
            
            # Получаем текущую историю из памяти
            current_memory = memory.load_memory_variables({})
            
            print(f"\nПосле сообщения {i+1}: '{message}'")
            
            # Выводим содержимое памяти (зависит от типа)
            if memory_name == "Summary Memory":
                print(f"Содержимое памяти: {current_memory['history']}")
            else:
                print(f"Количество сообщений в памяти: {len(current_memory['history'])}")
                for msg in current_memory['history']:
                    print(f"- {msg.type}: {msg.content[:30]}..." if len(msg.content) > 30 else f"- {msg.type}: {msg.content}")
        
        # Сохраняем конечное состояние памяти
        results[memory_name] = memory.load_memory_variables({})
    
    return results

# Запускаем функцию
print("\n=== 2.2 Типы памяти в LangChain ===")
memory_types_results = demonstrate_memory_types()

# %% [markdown]
# ### Сравнение типов памяти:
# 
# 1. **ConversationBufferMemory**
#    * Хранит полную историю сообщений
#    * Прост в использовании и понимании
#    * Проблема: потенциально может превысить контекстное окно модели при длинных разговорах
#    
# 2. **ConversationBufferWindowMemory**
#    * Хранит только фиксированное количество последних сообщений
#    * Гарантирует, что не превысим контекстное окно
#    * Проблема: может потерять важный контекст из более ранних сообщений
#    
# 3. **ConversationTokenBufferMemory**
#    * Ограничивает историю по количеству токенов, а не сообщений
#    * Более точный контроль над размером контекста
#    * Требует доступа к функции подсчета токенов LLM
#    
# 4. **ConversationSummaryMemory**
#    * Создает сводку разговора вместо хранения всех сообщений
#    * Эффективно сжимает длинные диалоги
#    * Может потерять специфические детали, но сохраняет общий контекст
#    
# Выбор типа памяти зависит от конкретного применения. В следующем разделе мы рассмотрим, как интегрировать память в диалоговую систему.

# %% [markdown]
# ## 3. Построение диалоговой системы с памятью
# 
# **💡 Ключевой концепт**: Мы можем легко интегрировать системы памяти в диалоговые приложения для создания более естественного взаимодействия.
# 
# В этом разделе мы построим простой чат-бот, который использует память для поддержания контекста разговора. Это позволит боту:
# * Запоминать информацию о пользователе
# * Ссылаться на предыдущие части разговора
# * Строить более связные и контекстуально релевантные ответы
# 
# **🔍 В этом разделе мы**:
# * Создадим интерактивную диалоговую систему с памятью
# * Интегрируем разные типы памяти
# * Разработаем интерфейс для тестирования диалоговой системы

# %%
def create_chatbot_with_memory(memory_type="buffer", memory_params=None):
    """
    Создает чат-бота с указанным типом памяти.
    
    Args:
        memory_type (str): Тип памяти ("buffer", "window", "token", "summary")
        memory_params (dict): Дополнительные параметры для памяти
    
    Returns:
        function: Функция чат-бота, принимающая сообщение пользователя
    """
    # Параметры по умолчанию
    if memory_params is None:
        memory_params = {}
    
    # Инициализируем модель
    model = ChatMistralAI(model="mistral-small-latest", temperature=0.7)
    
    # Выбираем тип памяти
    if memory_type == "buffer":
        memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    elif memory_type == "window":
        k = memory_params.get("k", 5)
        memory = ConversationBufferWindowMemory(k=k, return_messages=True, memory_key="history")
    elif memory_type == "token":
        max_token_limit = memory_params.get("max_token_limit", 500)
        memory = ConversationTokenBufferMemory(
            llm=model, max_token_limit=max_token_limit, return_messages=True, memory_key="history"
        )
    elif memory_type == "summary":
        memory = ConversationSummaryMemory(llm=model, return_messages=True, memory_key="history")
    else:
        raise ValueError(f"Неподдерживаемый тип памяти: {memory_type}")
    
    # Создаем промпт с плейсхолдером для истории
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - дружелюбный и полезный ассистент по имени Алиса. 
        
        У вас следующие характеристики:
        1. Вы отвечаете кратко и информативно
        2. Вы стараетесь помочь пользователю решить его задачу
        3. Вы используете информацию из предыдущих сообщений для контекста
        4. Вы можете поддерживать разговор на различные темы
        
        Имейте в виду пользовательские предпочтения и отвечайте в дружелюбном тоне.
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Создаем цепочку
    chain = prompt | model | StrOutputParser()
    
    # Функция чат-бота
    def chatbot(user_input):
        # Загружаем текущую историю из памяти
        memory_vars = memory.load_memory_variables({})
        
        # Вызываем модель
        response = chain.invoke({
            "history": memory_vars.get("history", []),
            "input": user_input
        })
        
        # Сохраняем взаимодействие в памяти
        memory.save_context({"input": user_input}, {"output": response})
        
        return response
    
    return chatbot, memory

def demonstrate_chatbot_with_memory():
    """
    Демонстрирует работу чат-бота с памятью в интерактивном режиме.
    """
    # Создаем бота с буферной памятью
    chatbot, memory = create_chatbot_with_memory(
        memory_type="buffer"
    )
    
    # Список предопределенных сообщений для тестирования
    test_conversation = [
        "Привет! Меня зовут Сергей.",
        "Я увлекаюсь программированием и хочу узнать больше о машинном обучении.",
        "Какие есть хорошие ресурсы для начинающих?",
        "Спасибо за рекомендации! А что насчет практических проектов?",
        "Интересно. Можешь ли ты напомнить, как меня зовут?",
        "А какую тему мы обсуждали изначально?"
    ]
    
    # Проводим разговор с ботом
    print("\nДемонстрация диалога с чат-ботом, использующим память:")
    
    for i, message in enumerate(test_conversation):
        print(f"\nСообщение {i+1}: {message}")
        response = chatbot(message)
        print(f"Ответ бота: {response}")
        
        # После каждого обмена показываем состояние памяти
        memory_state = memory.load_memory_variables({})
        print(f"\nСостояние памяти (количество сообщений): {len(memory_state.get('history', []))}")
    
    # Возвращаем финальное состояние памяти
    return memory.load_memory_variables({})

# Запускаем демонстрацию
print("\n=== 3. Построение диалоговой системы с памятью ===")
chatbot_memory_state = demonstrate_chatbot_with_memory()

# %%
def demonstrate_memory_comparison():
    """
    Сравнивает поведение чат-бота с разными типами памяти на одном и том же сценарии.
    """
    # Создаем тестовый сценарий - длинный разговор
    test_messages = [
        "Привет! Я Мария, мне 30 лет и я работаю инженером в IT-компании.",
        "Я ищу рекомендации по изучению искусственного интеллекта и машинного обучения.",
        "Какие языки программирования лучше всего подходят для работы с AI?",
        "А какие библиотеки самые популярные для глубокого обучения?",
        "Спасибо за информацию! Есть ли хорошие онлайн-курсы для начинающих?",
        "Отлично, я посмотрю эти ресурсы. Теперь я хочу сменить тему и поговорить о путешествиях.",
        "Какие интересные места можно посетить в Европе летом?",
        "А что насчет бюджетных вариантов путешествия?",
        "Интересно! А теперь, можешь ли ты напомнить, как меня зовут и чем я занимаюсь?",
        "А какие библиотеки для машинного обучения ты мне рекомендовал ранее?"
    ]
    
    # Типы памяти для сравнения
    memory_configs = [
        ("buffer", {}, "Стандартная буферная память"),
        ("window", {"k": 3}, "Память с окном в 3 сообщения"),
        ("token", {"max_token_limit": 300}, "Память с ограничением в 300 токенов"),
        ("summary", {}, "Суммаризирующая память")
    ]
    
    # Тестируем каждый тип памяти
    results = {}
    
    for memory_type, params, description in memory_configs:
        print(f"\n\n=== Тестирование: {description} ===\n")
        
        # Создаем бота с указанным типом памяти
        chatbot, memory = create_chatbot_with_memory(memory_type=memory_type, memory_params=params)
        
        # Проводим разговор
        responses = []
        for i, message in enumerate(test_messages):
            print(f"Сообщение {i+1}: {message}")
            response = chatbot(message)
            responses.append(response)
            print(f"Ответ: {response}\n")
            
            # Если это последний вопрос про вспоминание деталей, сохраняем ответ
            if i == len(test_messages) - 1:
                results[description] = response
    
    # Сравниваем как разные типы памяти справились с задачей вспоминания информации
    print("\n\n=== Сравнение эффективности памяти ===\n")
    for desc, response in results.items():
        print(f"{desc}:\n{response}\n")
    
    return results

# Запускаем сравнение
print("\n=== 3.2 Сравнение эффективности разных типов памяти ===")
memory_comparison_results = demonstrate_memory_comparison()

# %% [markdown]
# ### Результаты тестирования разных типов памяти:
# 
# Мы видим, что разные типы памяти имеют различную эффективность в зависимости от задачи:
# 
# * **Стандартная буферная память** хорошо работает при коротких разговорах, но может стать проблематичной при длинных беседах из-за ограничений контекстного окна модели.
# 
# * **Память с окном** хорошо запоминает недавние взаимодействия, но теряет информацию из начала разговора. Это может быть проблемой, если важная информация (например, имя пользователя) была упомянута давно.
# 
# * **Память с ограничением токенов** обеспечивает баланс между сохранением важной информации и ограничением размера контекста, но требует дополнительных вычислений для подсчета токенов.
# 
# * **Суммаризирующая память** хорошо сохраняет ключевую информацию даже из длинных диалогов, но может терять конкретные детали и требует дополнительных вызовов модели для создания сводок.
# 
# В следующем разделе мы рассмотрим, как использовать LangGraph для построения более сложных диалоговых систем с управлением состоянием.

# %% [markdown]
# ## 4. Управление состоянием диалога с помощью LangGraph
# 
# **💡 Ключевой концепт**: LangGraph позволяет создавать диалоговые системы со сложной логикой и управлением состоянием.
# 
# В предыдущих разделах мы работали с линейными диалоговыми системами, где каждое сообщение пользователя приводило к одному ответу модели. В реальных приложениях диалоговые системы часто должны:
# * Сохранять структурированное состояние диалога
# * Выполнять различные действия в зависимости от контекста
# * Переходить между разными режимами диалога
# 
# LangGraph предоставляет инструменты для создания таких сложных диалоговых систем.
# 
# **🔍 В этом разделе мы**:
# * Познакомимся с использованием StateGraph для создания диалоговых систем
# * Реализуем простой граф для обработки сообщений 
# * Научимся сохранять состояние диалога между запросами

# %%
def demonstrate_simple_stategraph():
    """
    Демонстрирует базовое использование StateGraph из LangGraph для создания
    простой диалоговой системы.
    """
    # Создаем модель
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    
    # Создаем граф с предопределенной схемой MessagesState
    workflow = StateGraph(state_schema=MessagesState)
    
    # Определяем функцию для узла чат-бота
    def call_model(state):
        """Узел, обрабатывающий сообщения пользователя и возвращающий ответ модели."""
        # Добавляем системное сообщение
        system_message = SystemMessage(content="""Вы - дружелюбный ассистент. Отвечайте кратко и информативно.
        Если пользователь прощается, тоже вежливо попрощайтесь.""")
        
        # Объединяем системное сообщение с историей сообщений
        messages = [system_message] + state["messages"]
        
        # Вызываем модель
        response = model.invoke(messages)
        
        # Возвращаем ответ в виде обновления состояния
        return {"messages": [response]}
    
    # Добавляем узел в граф
    workflow.add_node("model", call_model)
    
    # Устанавливаем связь между узлами (START ведет к модели)
    workflow.add_edge(START, "model")
    
    # Компилируем граф
    graph = workflow.compile()
    
    # Тестируем граф
    print("\nДемонстрация простого графа диалога с MessagesState:")
    
    # Начальное состояние - пустой список сообщений
    state = {"messages": []}
    
    # Последовательность сообщений для теста
    test_messages = [
        "Привет! Как ты можешь мне помочь?",
        "Расскажи мне о машинном обучении кратко.",
        "Спасибо за информацию! До свидания."
    ]
    
    # Обрабатываем каждое сообщение
    for message in test_messages:
        print(f"\nПользователь: {message}")
        
        # Добавляем сообщение пользователя в состояние
        state["messages"].append(HumanMessage(content=message))
        
        # Вызываем граф
        state = graph.invoke(state)
        
        # Выводим ответ модели
        last_message = state["messages"][-1]
        print(f"Ассистент: {last_message.content}")
    
    # Демонстрация сохранения состояния между вызовами
    print("\nДемонстрация сохранения контекста:")
    state = {"messages": []}
    
    # Первый вызов
    state["messages"].append(HumanMessage(content="Привет! Меня зовут Дмитрий."))
    state = graph.invoke(state)
    print(f"Пользователь: Привет! Меня зовут Дмитрий.")
    print(f"Ассистент: {state['messages'][-1].content}")
    
    # Второй вызов - проверяем, помнит ли бот имя
    state["messages"].append(HumanMessage(content="Как меня зовут?"))
    state = graph.invoke(state)
    print(f"Пользователь: Как меня зовут?")
    print(f"Ассистент: {state['messages'][-1].content}")
    
    return state

# Запускаем демонстрацию
print("\n=== 4. Простой граф диалога в LangGraph ===")
simple_graph_state = demonstrate_simple_stategraph()

# %% [markdown]
# ### Преимущества использования LangGraph для диалоговых систем:
# 
# 1. **Структурированное состояние**:
#    * Возможность хранить и обновлять сложное состояние диалога
#    * Четкое разделение между разными типами информации
# 
# 2. **Модульность**:
#    * Разделение сложной логики диалога на отдельные узлы
#    * Возможность легко добавлять новые функциональности без переписывания всей системы
# 
# 3. **Предсказуемость**:
#    * Четкая структура графа делает поведение системы более предсказуемым
#    * Легче отлаживать и тестировать отдельные компоненты
# 
# 4. **Сохранение контекста**:
#    * Автоматическое сохранение истории сообщений
#    * Легкая передача контекста между вызовами
# 
# В нашем примере мы построили простой граф диалога, который демонстрирует способность системы запоминать информацию о пользователе между запросами. Для более сложных систем LangGraph позволяет создавать графы с условными переходами, использовать различные типы узлов и сохранять более сложное структурированное состояние.

# %% [markdown]
# ## 5. Сохранение состояния графа (Checkpointing)
# 
# **💡 Ключевой концепт**: LangGraph предоставляет механизм для сохранения и восстановления состояния графа между сессиями.
# 
# Для создания полноценной диалоговой системы необходимо обеспечить персистентность - сохранение состояния между сеансами взаимодействия с пользователем. LangGraph предлагает встроенный механизм **checkpointing**, который позволяет сохранять состояние графа и восстанавливать его при необходимости.
# 
# **🔍 В этом разделе мы**:
# * Рассмотрим механизм сохранения состояния с использованием checkpointer
# * Реализуем простое хранилище состояний в памяти
# * Создадим диалоговую систему с персистентным состоянием
# 
# ### Использование MemorySaver для сохранения состояния

# %%
from langgraph.checkpoint.memory import MemorySaver

def demonstrate_checkpointing():
    """
    Демонстрирует использование checkpointer для сохранения состояния графа.
    """
    # Создаем простой граф с MessagesState
    model = ChatMistralAI(model="mistral-small-latest", temperature=0)
    
    # Создаем граф
    workflow = StateGraph(state_schema=MessagesState)
    
    # Определяем функцию для обработки сообщений
    def call_model(state):
        """Узел для обработки сообщений."""
        # Создаем системное сообщение
        system_message = SystemMessage(content="""Вы - ассистент с памятью. 
        Используйте информацию из предыдущих сообщений для ответов.
        Отвечайте кратко и информативно.""")
        
        # Объединяем системное сообщение с историей
        messages = [system_message] + state["messages"]
        
        # Вызываем модель
        response = model.invoke(messages)
        
        # Возвращаем обновленное состояние
        return {"messages": [response]}
    
    # Добавляем узел в граф
    workflow.add_node("model", call_model)
    
    # Устанавливаем связь между узлами
    workflow.add_edge(START, "model")
    
    # Создаем хранилище состояний в памяти
    memory_saver = MemorySaver()
    
    # Компилируем граф с использованием checkpointer
    graph = workflow.compile(checkpointer=memory_saver)
    
    print("\nДемонстрация сохранения состояния с помощью checkpointer:")
    
    # ID для первого пользователя
    user1_thread_id = "user1"
    
    # Создаем config для первого пользователя
    config_user1 = {"configurable": {"thread_id": user1_thread_id}}
    
    # Начальное состояние для первого пользователя
    state_user1 = {"messages": []}
    
    # Добавляем сообщение от первого пользователя
    state_user1["messages"].append(HumanMessage(content="Привет! Меня зовут Анна."))
    
    # Вызываем граф с сохранением состояния
    state_user1 = graph.invoke(state_user1, config=config_user1)
    
    # Выводим ответ
    print(f"Пользователь 1: Привет! Меня зовут Анна.")
    print(f"Ассистент -> Пользователь 1: {state_user1['messages'][-1].content}")
    
    # ID для второго пользователя
    user2_thread_id = "user2"
    
    # Создаем config для второго пользователя
    config_user2 = {"configurable": {"thread_id": user2_thread_id}}
    
    # Начальное состояние для второго пользователя
    state_user2 = {"messages": []}
    
    # Добавляем сообщение от второго пользователя
    state_user2["messages"].append(HumanMessage(content="Здравствуйте! Я Иван."))
    
    # Вызываем граф с сохранением состояния для второго пользователя
    state_user2 = graph.invoke(state_user2, config=config_user2)
    
    # Выводим ответ
    print(f"\nПользователь 2: Здравствуйте! Я Иван.")
    print(f"Ассистент -> Пользователь 2: {state_user2['messages'][-1].content}")
    
    # Добавляем второе сообщение от первого пользователя
    state_user1["messages"].append(HumanMessage(content="Как меня зовут?"))
    
    # Вызываем граф с сохранением состояния для продолжения диалога
    state_user1 = graph.invoke(state_user1, config=config_user1)
    
    # Выводим ответ
    print(f"\nПользователь 1: Как меня зовут?")
    print(f"Ассистент -> Пользователь 1: {state_user1['messages'][-1].content}")
    
    # Добавляем второе сообщение от второго пользователя
    state_user2["messages"].append(HumanMessage(content="Как меня зовут?"))
    
    # Вызываем граф с сохранением состояния для продолжения диалога
    state_user2 = graph.invoke(state_user2, config=config_user2)
    
    # Выводим ответ
    print(f"\nПользователь 2: Как меня зовут?")
    print(f"Ассистент -> Пользователь 2: {state_user2['messages'][-1].content}")
    
    # Демонстрация восстановления состояния
    print("\nДемонстрация восстановления состояния из checkpointer:")
    
    # Правильный способ получения состояния графа - используем graph.get_state
    # Получаем сохраненное состояние для первого пользователя
    state_snapshot = graph.get_state(config_user1)
    
    if state_snapshot:
        print(f"Получено состояние графа для Пользователя 1")
        if "messages" in state_snapshot.values:
            print(f"Количество сообщений в сохраненном состоянии: {len(state_snapshot.values['messages'])}")
    
    # Эмулируем новую сессию с восстановлением состояния
    print("\nЭмуляция новой сессии с восстановлением состояния:")
    
    # Восстанавливаем состояние напрямую из снапшота состояния
    restored_state = state_snapshot.values if state_snapshot else {"messages": []}
    
    # Добавляем новое сообщение
    restored_state["messages"].append(HumanMessage(content="Что мы обсуждали ранее?"))
    
    # Вызываем граф с восстановленным состоянием
    result_state = graph.invoke(restored_state, config=config_user1)
    
    # Выводим ответ
    print(f"Пользователь 1: Что мы обсуждали ранее?")
    print(f"Ассистент -> Пользователь 1: {result_state['messages'][-1].content}")
    
    return memory_saver

# Запускаем демонстрацию
print("\n=== 5. Сохранение состояния графа (Checkpointing) ===")
memory_checkpointer = demonstrate_checkpointing()

# %% [markdown]
# ### Расширенные возможности сохранения состояния
# 
# В реальных приложениях вместо `MemorySaver` (который хранит состояния только в памяти) часто используются более надежные хранилища:
# 
# 1. **Redis**:
#    ```python
#    from langgraph.checkpoint.redis import RedisSaver
#    
#    redis_saver = RedisSaver(url="redis://localhost:6379/0")
#    graph = workflow.compile(checkpointer=redis_saver)
#    ```
# 
# 2. **Настраиваемые сохранялки**:
#    Вы можете создать свой класс checkpointer, реализовав методы `put()`, `get()` и `delete()`.
#    
# ### Преимущества использования checkpointing:
# 
# * **Многопользовательская поддержка** - возможность обрабатывать диалоги с множеством пользователей
# * **Персистентность** - сохранение состояния между запусками приложения
# * **Масштабируемость** - возможность распределять нагрузку между несколькими экземплярами приложения
# * **Восстановление после сбоев** - возможность продолжить диалог с последней сохраненной точки
# 
# Это особенно важно для производственных приложений, где требуется надежность и возможность масштабирования.

# %% [markdown]
# ## Заключение
# 
# **🎓 Что вы узнали в этой лаборатории**:
# 
# 1. **🧩 Базовые принципы памяти** - как сохранять контекст разговора для более осмысленного взаимодействия
# 2. **🔄 Типы памяти в LangChain** - различные реализации систем памяти и их преимущества
# 3. **🚀 Управление состоянием** - как структурировать и обновлять состояние диалоговой системы с помощью LangGraph
# 4. **📊 Потоки сообщений** - создание диалоговых систем с сохранением контекста между запросами
# 5. **💾 Сохранение состояния** - использование checkpointing для персистентности диалоговых систем
