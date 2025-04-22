from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
import orjson

from bot.config.settings import Settings


def create_bot(settings: Settings):
    session: AiohttpSession = AiohttpSession(
        json_loads=orjson.loads,
        json_dumps=lambda obj, **kwargs: orjson.dumps(obj).decode('utf-8')
    )
    return Bot(
        token=settings.bot.token,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML
        ),
        session=session
    )
