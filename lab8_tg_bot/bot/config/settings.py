import os
from dataclasses import field, dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class BotSettings:
    token: str = field(default_factory=lambda: os.getenv("BOT_TOKEN"))


@dataclass
class Settings:
    bot: BotSettings = field(default_factory=BotSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        env_name = ".env.dev" if os.getenv("BOT_MODE") == "dev" else ".env"
        env_path = Path(f"{os.curdir}/{env_name}")
        if env_path.is_file():
            from dotenv import load_dotenv

            load_dotenv(env_path)
        return Settings()
    

@lru_cache(maxsize=1, typed=True)
def get_settings() -> Settings:
    return Settings.from_env()
