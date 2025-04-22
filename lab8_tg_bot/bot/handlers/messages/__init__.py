from typing import Final

from aiogram import Router

from .start import router as start_router


router: Final[Router] = Router(name=__name__)
router.include_routers(
    start_router
)
