import os
from os import getenv
from pathlib import Path

from aiogram import Bot, Dispatcher, F, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from models import load_ste, load_stt

TOKEN = getenv("BOT_TOKEN")
if TOKEN is None:
    raise RuntimeError("`BOT_TOKEN` is not set.")
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message):
    assert message.from_user is not None
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


@dp.message(F.voice | F.audio)
async def audio_handler(message: Message):
    if message.voice:
        file_id = message.voice.file_id
        filename = message.voice.file_unique_id
    else:
        assert message.audio is not None
        file_id = message.audio.file_id
        filename = message.audio.file_unique_id

    file = await bot.get_file(file_id)

    assert file.file_path is not None

    extension = Path(file.file_path).suffix
    path = Path("telegram/voices/" + filename + extension)
    await bot.download_file(file.file_path, destination=path)

    stt_model = load_stt()

    transcription = stt_model.transcribe_longform(path)
    transcription = " ".join([utt["transcription"] for utt in transcription])
    await message.answer(transcription.strip())
    os.remove(path)


async def start_bot():
    await dp.start_polling(bot)
