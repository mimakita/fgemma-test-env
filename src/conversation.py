"""Main conversation loop - orchestrates gemma3:4b dialogue and FunctionGemma routing."""

import logging
import sys

from src.config import (
    CONVERSATION_MODEL,
    CONVERSATION_OPTIONS,
    CONVERSATION_KEEP_ALIVE,
)
from src.ollama_client import OllamaClient
from src.functions import init_all, registry
from src.router import FunctionRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages the main conversation loop with function routing."""

    def __init__(self):
        self.client = OllamaClient()
        init_all()
        self.router = FunctionRouter(self.client, registry)
        self.history: list[dict] = []
        self.system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Respond naturally in the same language "
                "as the user. Keep responses concise and helpful."
            ),
        }

    def run(self):
        """Main interactive loop."""
        print("=" * 50)
        print("FuncGemma Chat")
        print("Models: gemma3:4b (conversation) + functiongemma (routing)")
        print("Type 'quit' to exit, 'history' to show conversation")
        print("=" * 50)

        registered = registry.get_all_names()
        print(f"Registered functions: {', '.join(registered)}")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if user_input.lower() == "history":
                self._show_history()
                continue

            # 1. Add user message to history
            self.history.append({"role": "user", "content": user_input})

            # 2. Get conversational response from gemma3:4b
            print("\n[Generating response...]")
            try:
                messages = [self.system_message] + self.history
                response = self.client.chat_completion(
                    model=CONVERSATION_MODEL,
                    messages=messages,
                    options=CONVERSATION_OPTIONS,
                    keep_alive=CONVERSATION_KEEP_ALIVE,
                )
                assistant_msg = response.message.content or ""
            except Exception as e:
                logger.error(f"Conversation model error: {e}")
                assistant_msg = f"[Error: Could not generate response - {e}]"

            self.history.append({"role": "assistant", "content": assistant_msg})
            print(f"\nAssistant: {assistant_msg}")

            # 3. Route through FunctionGemma
            print("\n[Routing through FunctionGemma...]")
            try:
                result = self.router.route(self.history)
            except Exception as e:
                logger.error(f"Router error: {e}")
                result = None

            if result and result.should_call:
                print(f"\n  >> Function: {result.function_name}")
                print(f"  >> Arguments: {result.arguments}")

                # Execute the function
                func_result = registry.execute(
                    result.function_name, result.arguments
                )
                print(f"  >> Result: {func_result}")
            else:
                print("  >> No function triggered")

            print()

    def _show_history(self):
        """Display conversation history."""
        if not self.history:
            print("[No conversation history]")
            return
        print("\n--- Conversation History ---")
        for i, msg in enumerate(self.history):
            role = msg["role"].capitalize()
            content = msg["content"][:200]
            print(f"  [{i}] {role}: {content}")
        print("---\n")


def main():
    manager = ConversationManager()
    manager.run()


if __name__ == "__main__":
    main()
