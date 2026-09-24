import os
import sys
import uuid
from typing import Any, Dict
from dotenv import load_dotenv
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from system_health_agent.backend import create_system_health_agent

load_dotenv()


def get_checkpointer():
    """Returns MongoDBSaver if configured & installed, otherwise falls back to MemorySaver."""
    mongodb_uri = os.getenv("MONGODB_URI")
    if mongodb_uri:
        try:
            from pymongo import MongoClient
            from langgraph.checkpoint.mongodb import MongoDBSaver

            client = MongoClient(mongodb_uri)
            checkpointer = MongoDBSaver(client=client, db_name="agent_health_db")
            print(" Connected to MongoDB Checkpointer (agent_health_db)")
            return checkpointer
        except Exception as e:
            print(f"⚠️ Could not initialize MongoDBSaver ({e}). Falling back to MemorySaver.")
    
    return MemorySaver()


def handle_interrupt(agent, config, interrupt_payload: Any):
    """Handles HITL decision prompt and returns the appropriate Command(resume=...)."""
    print("\n" + "=" * 60)
    print("⚠️  INTERRUPT: Tool execution requires human approval!")
    print("=" * 60)

    # Payload typically has action_requests
    action_requests = getattr(interrupt_payload, "action_requests", None)
    if action_requests is None and isinstance(interrupt_payload, dict):
        action_requests = interrupt_payload.get("action_requests", [])

    tool_call_name = "inspect_directory_metadata"
    tool_args = {}
    if action_requests:
        req = action_requests[0]
        tool_call_name = req.get("name", tool_call_name) if isinstance(req, dict) else getattr(req, "name", tool_call_name)
        tool_args = req.get("args", {}) if isinstance(req, dict) else getattr(req, "args", {})

    print(f"Tool: {tool_call_name}")
    print(f"Arguments: {tool_args}")
    print("-" * 60)
    print("Options:")
    print("  [1] Approve - Run tool with current arguments")
    print("  [2] Edit    - Modify directory path arguments")
    print("  [3] Reject  - Cancel this tool call")
    print("-" * 60)

    choice = input("Enter decision (1/2/3 or approve/edit/reject) [default: 1]: ").strip().lower()

    if choice in ("2", "edit"):
        current_path = tool_args.get("dir_path", "./src")
        new_path = input(f"Enter new dir_path (current: {current_path}): ").strip()
        if not new_path:
            new_path = current_path
        
        decision = {
            "type": "edit",
            "edited_action": {
                "name": tool_call_name,
                "args": {"dir_path": new_path},
            },
        }
        print(f" Overriding path to: {new_path}")
    elif choice in ("3", "reject"):
        reason = input("Enter rejection reason (optional): ").strip()
        decision = {
            "type": "reject",
            "message": reason or "Action rejected by operator",
        }
        print(" Action rejected.")
    else:
        decision = {"type": "approve"}
        print(" Approved tool execution.")

    return Command(resume={"decisions": [decision]})


def run_cli_session():
    """Runs interactive CLI session for the System Health Agent."""
    print("=" * 60)
    print("🩺 System Health & Diagnostic Agent with HITL & Summarization")
    print("=" * 60)

    checkpointer = get_checkpointer()
    agent = create_system_health_agent(
        model_name="gemini-2.5-flash",
        checkpointer=checkpointer,
        summarization_trigger_messages=6,
    )

    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}}
    print(f"Active Session Thread ID: {thread_id}")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("Operator > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Session terminated. Goodbye!")
                break

            current_input: Any = {"messages": [{"role": "user", "content": user_input}]}

            # Processing loop (handles possible sequential interrupts & resume)
            while current_input is not None:
                interrupt_found = False

                for event in agent.stream(current_input, config=config, stream_mode="updates"):
                    # Check for __interrupt__
                    if "__interrupt__" in event:
                        interrupt_data = event["__interrupt__"][0]
                        # Prompt operator and generate Command(resume=...)
                        resume_command = handle_interrupt(agent, config, interrupt_data.value)
                        current_input = resume_command
                        interrupt_found = True
                        break

                    # Print assistant message updates
                    for node_name, state_update in event.items():
                        messages = state_update.get("messages", [])
                        for msg in messages:
                            content = getattr(msg, "content", None)
                            role = getattr(msg, "type", "message")
                            if role == "ai" and content:
                                print(f"\nAgent > {content}\n")

                if not interrupt_found:
                    current_input = None

        except (KeyboardInterrupt, EOFError):
            print("\nExiting session...")
            break
        except Exception as exc:
            print(f"\n❌ Error during execution: {exc}\n")


if __name__ == "__main__":
    run_cli_session()
