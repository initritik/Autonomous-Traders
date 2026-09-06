from .traders import Trader, GEMINI_MODEL, DEEPSEEK_MODEL, GROQ_MODEL
from typing import List
import asyncio
from .tracers import LogTracer
from agents import add_trace_processor
from .market import is_market_open
from dotenv import load_dotenv
import os

load_dotenv(override=True)

RUN_EVERY_N_MINUTES = int(os.getenv("RUN_EVERY_N_MINUTES", "60"))
RUN_EVEN_WHEN_MARKET_IS_CLOSED = (
    os.getenv("RUN_EVEN_WHEN_MARKET_IS_CLOSED", "false").strip().lower() == "true"
)
USE_MANY_MODELS = os.getenv("USE_MANY_MODELS", "false").strip().lower() == "true"

names = ["Warren", "George", "Ray", "Cathie"]
lastnames = ["Patience", "Bold", "Systematic", "Crypto"]

if USE_MANY_MODELS:
    # All three providers are reached through the OpenAI SDK (see traders.py):
    # Gemini and Groq via their OpenAI-compatible endpoints, DeepSeek likewise.
    # We only have 3 free providers for 4 traders, so Gemini - our primary
    # model - is doubled up (Warren and Cathie).
    model_names = [
        GEMINI_MODEL,    # Warren  -> Google Gemini
        DEEPSEEK_MODEL,  # George  -> DeepSeek
        GROQ_MODEL,      # Ray     -> Groq (Llama 3.3 70B)
        GEMINI_MODEL,    # Cathie  -> Google Gemini
    ]
    short_model_names = [
        "Gemini 3.1 Flash Lite",
        "DeepSeek Chat",
        "Groq Llama 3.3 70B",
        "Gemini 3.1 Flash Lite",
    ]
else:
    # Single-model mode: every trader runs on Gemini 3.1 Flash Lite via the
    # OpenAI SDK pointed at Google's OpenAI-compatible base URL.
    model_names = [GEMINI_MODEL] * 4
    short_model_names = ["Gemini 3.1 Flash Lite"] * 4


def create_traders() -> List[Trader]:
    traders = []
    for name, lastname, model_name in zip(names, lastnames, model_names):
        traders.append(Trader(name, lastname, model_name))
    return traders


async def run_every_n_minutes():
    add_trace_processor(LogTracer())
    traders = create_traders()
    while True:
        if RUN_EVEN_WHEN_MARKET_IS_CLOSED or is_market_open():
            await asyncio.gather(*[trader.run() for trader in traders])
        else:
            print("Market is closed, skipping run")
        await asyncio.sleep(RUN_EVERY_N_MINUTES * 60)


if __name__ == "__main__":
    print(f"Starting scheduler to run every {RUN_EVERY_N_MINUTES} minutes")
    asyncio.run(run_every_n_minutes())