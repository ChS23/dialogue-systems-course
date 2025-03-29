import sys
import asyncio

from bot.config.settings import get_settings
from bot.factory.bot import create_bot
from bot.factory.dispatcher import create_dispatcher


async def main():
    settings = get_settings()

    bot = create_bot(settings)

    dp = await create_dispatcher(settings)
    await bot.delete_webhook(drop_pending_updates=True)

    await dp.start_polling(bot)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    else:
        import uvloop
        if sys.version_info >= (3, 11):
            with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
                runner.run(main())
        else:
            uvloop.install()
            asyncio.run(main())
