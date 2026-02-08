import asyncio
import logging
import sys

import telegram.bot as bot

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(bot.start_bot())
