from aiogram import Dispatcher

from bot.handlers import messages


def setup_routers(dp: Dispatcher):
    dp.include_routers(
        messages.router
    )


__all__ = "setup_routers"