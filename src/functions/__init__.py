"""Function registry initialization."""

from src.functions.registry import FunctionRegistry, FunctionDefinition

# Global registry
registry = FunctionRegistry()


def init_all():
    """Register all functions with the global registry."""
    from src.functions import (
        travel,
        celebrity,
        shopping,
        sentiment,
        weather,
        schedule,
        translation,
    )

    travel.register(registry)
    celebrity.register(registry)
    shopping.register(registry)
    sentiment.register(registry)
    weather.register(registry)
    schedule.register(registry)
    translation.register(registry)
