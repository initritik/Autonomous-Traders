from contextlib import AsyncExitStack
from .accounts_client import read_accounts_resource, read_strategy_resource
from .tracers import make_trace_id
from agents import Agent, Tool, Runner, OpenAIChatCompletionsModel, trace
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import json
from .templates import (
    researcher_instructions,
    trader_instructions,
    trade_message,
    rebalance_message,
    research_tool,
)
from .mcp_servers import trader_mcp_servers, researcher_mcp_servers

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Provider API keys.
#
# We deliberately do NOT use a native OpenAI key/client anywhere in this
# project. Instead we use the `openai` Python SDK purely as a generic,
# OpenAI-compatible HTTP client, and point it at three different free/low-cost
# providers by overriding `base_url`. Each provider (Google Gemini, DeepSeek,
# Groq) exposes an OpenAI-compatible `/chat/completions` endpoint, so the SDK
# (and the OpenAI Agents SDK's `OpenAIChatCompletionsModel`) work unmodified.
# ---------------------------------------------------------------------------
google_api_key = os.getenv("GOOGLE_API_KEY")      # Gemini API key (aistudio.google.com)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")  # DeepSeek API key (platform.deepseek.com)
groq_api_key = os.getenv("GROQ_API_KEY")          # Groq API key (console.groq.com) - free tier

# OpenAI-compatible base URLs for each provider.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Default / fallback model ids per provider. Gemini 3.1 Flash Lite is our
# primary model across the codebase; DeepSeek and Groq are used to give the
# four traders genuinely different "brains" when USE_MANY_MODELS=true.
GEMINI_MODEL = "gemini-3.1-flash-lite"
DEEPSEEK_MODEL = "deepseek-chat"
GROQ_MODEL = "llama-3.3-70b-versatile"

MAX_TURNS = 30

# One AsyncOpenAI client per provider. Same SDK class throughout - only the
# base_url + api_key change. This is the "openai sdk with a different
# base_url" trick that lets any OpenAI-compatible provider stand in for
# OpenAI itself.
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)


def get_model(model_name: str) -> OpenAIChatCompletionsModel:
    """Map a model name to the right OpenAI-compatible client.

    We route purely by matching keywords in the model name, since that's all
    the rest of the codebase (trading_floor.py) passes around. Every branch
    returns an `OpenAIChatCompletionsModel` wired to the OpenAI SDK client for
    that provider - there is no native OpenAI branch, by design.
    """
    if "gemini" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=gemini_client)
    elif "deepseek" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=deepseek_client)
    elif "llama" in model_name or "groq" in model_name or "gemma" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=groq_client)
    else:
        # Fail fast rather than silently falling back to a native OpenAI
        # call we never intended to make (no OPENAI_API_KEY is configured).
        raise ValueError(
            f"Unrecognized model_name '{model_name}'. Expected a Gemini, "
            "DeepSeek, or Groq model id (see GEMINI_MODEL / DEEPSEEK_MODEL / "
            "GROQ_MODEL in traders.py)."
        )


async def get_researcher(mcp_servers, model_name) -> Agent:
    researcher = Agent(
        name="Researcher",
        instructions=researcher_instructions(),
        model=get_model(model_name),
        mcp_servers=mcp_servers,
    )
    return researcher


async def get_researcher_tool(mcp_servers, model_name) -> Tool:
    researcher = await get_researcher(mcp_servers, model_name)
    return researcher.as_tool(tool_name="Researcher", tool_description=research_tool())


class Trader:
    def __init__(self, name: str, lastname="Trader", model_name=GEMINI_MODEL):
        self.name = name
        self.lastname = lastname
        self.agent = None
        self.model_name = model_name
        self.do_trade = True

    async def create_agent(self, trader_mcp_servers, researcher_mcp_servers) -> Agent:
        tool = await get_researcher_tool(researcher_mcp_servers, self.model_name)
        self.agent = Agent(
            name=self.name,
            instructions=trader_instructions(self.name),
            model=get_model(self.model_name),
            tools=[tool],
            mcp_servers=trader_mcp_servers,
        )
        return self.agent

    async def get_account_report(self) -> str:
        account = await read_accounts_resource(self.name)
        account_json = json.loads(account)
        account_json.pop("portfolio_value_time_series", None)
        return json.dumps(account_json)

    async def run_agent(self, trader_mcp_servers, researcher_mcp_servers):
        self.agent = await self.create_agent(trader_mcp_servers, researcher_mcp_servers)
        account = await self.get_account_report()
        strategy = await read_strategy_resource(self.name)
        message = (
            trade_message(self.name, strategy, account)
            if self.do_trade
            else rebalance_message(self.name, strategy, account)
        )
        await Runner.run(self.agent, message, max_turns=MAX_TURNS)

    async def run_with_mcp_servers(self):
        async with AsyncExitStack() as stack:
            trader_servers = [
                await stack.enter_async_context(server) for server in trader_mcp_servers()
            ]
            researcher_servers = [
                await stack.enter_async_context(server)
                for server in researcher_mcp_servers(self.name)
            ]
            await self.run_agent(trader_servers, researcher_servers)

    async def run_with_trace(self):
        trace_name = f"{self.name}-trading" if self.do_trade else f"{self.name}-rebalancing"
        trace_id = make_trace_id(f"{self.name.lower()}")
        with trace(trace_name, trace_id=trace_id):
            await self.run_with_mcp_servers()

    async def run(self):
        try:
            await self.run_with_trace()
        except Exception as e:
            print(f"Error running trader {self.name}: {e}")
        self.do_trade = not self.do_trade