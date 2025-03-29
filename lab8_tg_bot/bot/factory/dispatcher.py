from aiogram import Dispatcher
from aiogram.utils.callback_answer import CallbackAnswerMiddleware

from bot.handlers import setup_routers


def _setup_inner_middlewares(dispatcher: Dispatcher) -> None:
    dispatcher.callback_query.middleware(CallbackAnswerMiddleware())


async def create_dispatcher(settings) -> Dispatcher:
    dispatcher: Dispatcher = Dispatcher(
        name="main"
    )
    setup_routers(dispatcher)
    _setup_inner_middlewares(dispatcher)

    return dispatcher
