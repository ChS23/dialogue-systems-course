from typing import Final

from aiogram import Router
from aiogram.filters import CommandStart, CommandObject
from aiogram.types import Message


router: Final[Router] = Router(name=__name__)


@router.message(CommandStart())
async def start_command(
        message: Message,
        command: CommandObject
):
    user_name = message.from_user.first_name
    
    assignment_text = f"""Привет, {user_name}! 👋

<b>Задание на контрольную работу:</b>

Разработать Telegram-бота с следующими функциями:

1️⃣ <b>Поиск информации в интернете:</b>
   - Бот должен уметь искать информацию по запросу пользователя

2️⃣ <b>Поиск по вашей доменной зоне:</b>
   - Реализовать функцию поиска из документов, загруженных ранее

3️⃣ <b>Общение с пользователем:</b>
   - Интерактивное меню с кнопками
   - Обработка текстовых команд
   - Поддержка inline-режима

<b>Технические требования:</b>
▫️ Использовать aiogram 3.x
▫️ Асинхронная архитектура
▫️ Хранение данных (БД на выбор)

Удачи! 🚀"""
    
    await message.answer(assignment_text)
