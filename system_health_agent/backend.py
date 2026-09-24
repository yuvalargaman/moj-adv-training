import os
from typing import Any, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    SummarizationMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver

from system_health_agent.tools import (
    get_system_metrics,
    check_endpoint_health,
    inspect_directory_metadata,
)

load_dotenv()

SYSTEM_PROMPT = """You are an Infrastructure Health & Diagnostic Specialist Agent.
Your role is to diagnose host performance, inspect endpoint availability, and examine filesystem structures safely.

### OPERATIONAL GUIDELINES:
1. Always follow step-by-step diagnostic reasoning.
2. For system performance queries, check real-time CPU, RAM, and Disk metrics.
3. For endpoint checks, analyze HTTP status codes, latency, and headers.
4. When inspecting local directories, explain what metadata you intend to inspect.
5. Provide clear, executive diagnostic summaries and actionable recommendations.
"""


def create_system_health_agent(
    model_name: str = "gemini-2.5-flash",
    checkpointer: Optional[Any] = None,
    summarization_trigger_messages: int = 6,
):
    """Factory creating the System Health Agent with HITL and Summarization middlewares."""
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
    )

    tools = [
        get_system_metrics,
        check_endpoint_health,
        inspect_directory_metadata,
    ]

    # HITL: interrupt on inspect_directory_metadata, auto-approve the others
    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "inspect_directory_metadata": True,
            "get_system_metrics": False,
            "check_endpoint_health": False,
        }
    )

    # Summarization: condense context when exceeding message threshold
    summarization_middleware = SummarizationMiddleware(
        model=llm,
        trigger=("messages", summarization_trigger_messages),
        keep=("messages", 2),
    )

    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            hitl_middleware,
            summarization_middleware,
        ],
        checkpointer=checkpointer,
    )

    return agent
