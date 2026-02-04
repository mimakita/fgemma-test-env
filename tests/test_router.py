"""Tests for the function router and registry."""

import pytest
from src.functions.registry import FunctionRegistry, FunctionDefinition
from src.router import FunctionRouter


# --- Registry Tests ---

def test_registry_register_and_get():
    reg = FunctionRegistry()
    func_def = FunctionDefinition(
        name="test_func",
        description="A test function",
        parameters={"type": "object", "properties": {}},
        handler=lambda: {"result": "ok"},
    )
    reg.register(func_def)
    assert reg.get("test_func") is func_def
    assert reg.get("nonexistent") is None


def test_registry_get_all_names():
    reg = FunctionRegistry()
    reg.register(FunctionDefinition("a", "desc", {}, lambda: None))
    reg.register(FunctionDefinition("b", "desc", {}, lambda: None))
    names = reg.get_all_names()
    assert "a" in names
    assert "b" in names
    assert len(names) == 2


def test_registry_get_all_tool_schemas():
    reg = FunctionRegistry()
    reg.register(
        FunctionDefinition(
            name="test",
            description="Test desc",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            handler=lambda x: {"x": x},
        )
    )
    schemas = reg.get_all_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "test"
    assert schemas[0]["function"]["description"] == "Test desc"


def test_registry_execute():
    reg = FunctionRegistry()
    reg.register(
        FunctionDefinition(
            name="add",
            description="Add numbers",
            parameters={},
            handler=lambda a, b: {"sum": a + b},
        )
    )
    result = reg.execute("add", {"a": 1, "b": 2})
    assert result == {"sum": 3}


def test_registry_execute_unknown():
    reg = FunctionRegistry()
    result = reg.execute("nonexistent", {})
    assert "error" in result


def test_registry_execute_bad_args():
    reg = FunctionRegistry()
    reg.register(
        FunctionDefinition("f", "desc", {}, lambda x: {"x": x})
    )
    result = reg.execute("f", {"y": 1})
    assert "error" in result


# --- Router Fallback Parser Tests ---

def test_router_parse_raw_json():
    reg = FunctionRegistry()
    reg.register(FunctionDefinition("travel_guide", "desc", {}, lambda **kw: kw))

    from src.ollama_client import OllamaClient
    router = FunctionRouter(OllamaClient(), reg)

    result = router._parse_raw_function_call('{"name": "travel_guide", "arguments": {"destination": "Tokyo"}}')
    assert result is not None
    name, args = result
    assert name == "travel_guide"
    assert args["destination"] == "Tokyo"


def test_router_parse_raw_empty():
    reg = FunctionRegistry()
    from src.ollama_client import OllamaClient
    router = FunctionRouter(OllamaClient(), reg)

    assert router._parse_raw_function_call("") is None
    assert router._parse_raw_function_call(None) is None
    assert router._parse_raw_function_call("just some text") is None


def test_router_parse_function_args():
    reg = FunctionRegistry()
    from src.ollama_client import OllamaClient
    router = FunctionRouter(OllamaClient(), reg)

    args = router._parse_function_args('destination="Tokyo", info_type="general"')
    assert args == {"destination": "Tokyo", "info_type": "general"}


# --- Function Init Test ---

def test_all_functions_register():
    from src.functions import init_all, registry

    # Reset registry
    registry._functions.clear()
    init_all()

    expected = [
        "travel_guide",
        "celebrity_info",
        "shopping_intent",
        "sentiment_label",
        "weather_info",
        "schedule_reminder",
        "translation_assist",
    ]
    for name in expected:
        assert registry.get(name) is not None, f"Function {name} not registered"

    assert len(registry.get_all_names()) == 7


def test_all_tool_schemas_valid():
    from src.functions import init_all, registry

    registry._functions.clear()
    init_all()

    schemas = registry.get_all_tool_schemas()
    assert len(schemas) == 7

    for schema in schemas:
        assert schema["type"] == "function"
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        assert func["parameters"]["type"] == "object"
        assert "properties" in func["parameters"]
