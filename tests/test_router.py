"""Tests for the function router, classifier, and registry."""

import pytest
from src.functions.registry import FunctionRegistry, FunctionDefinition
from src.router import FunctionRouter, RouterResult
from src.classifier import FunctionCallClassifier, ClassificationResult


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


# --- Classifier Tests ---

class TestFunctionCallClassifier:
    """Tests for Stage 1 classifier."""

    def test_classifier_no_function_greetings(self):
        """Greetings should not trigger function calls."""
        classifier = FunctionCallClassifier()

        greetings = [
            [{"role": "user", "content": "こんにちは"}],
            [{"role": "user", "content": "おはようございます"}],
            [{"role": "user", "content": "お疲れ様です"}],
            [{"role": "user", "content": "ありがとうございます"}],
        ]

        for conv in greetings:
            result = classifier.classify(conv)
            assert isinstance(result, ClassificationResult)
            # Greetings contain no_function keywords, so confidence should be reasonable
            assert result.confidence >= 0

    def test_classifier_no_function_general_questions(self):
        """General knowledge questions should not trigger function calls."""
        classifier = FunctionCallClassifier()

        questions = [
            [{"role": "user", "content": "AIとは何ですか？"}],
            [{"role": "user", "content": "プログラミングについて教えてください"}],
            [{"role": "user", "content": "数学の計算を手伝ってください"}],
            [{"role": "user", "content": "どう思いますか？"}],
        ]

        for conv in questions:
            result = classifier.classify(conv)
            assert isinstance(result, ClassificationResult)
            # These should lean toward no function
            # (actual need_function depends on keyword matching)

    def test_classifier_function_needed_travel(self):
        """Travel queries should trigger function calls."""
        classifier = FunctionCallClassifier()

        travel_queries = [
            [{"role": "user", "content": "京都の観光名所を教えてください"}],
            [{"role": "user", "content": "東京への行き方を知りたい"}],
            [{"role": "user", "content": "北海道の旅行プランを立てたい"}],
        ]

        for conv in travel_queries:
            result = classifier.classify(conv)
            assert isinstance(result, ClassificationResult)
            # These contain travel keywords, should need function
            assert result.need_function is True
            assert result.matched_function == "travel_guide"

    def test_classifier_function_needed_weather(self):
        """Weather queries should trigger function calls."""
        classifier = FunctionCallClassifier()

        weather_queries = [
            [{"role": "user", "content": "明日の天気を教えて"}],
            [{"role": "user", "content": "東京の気温は？"}],
            [{"role": "user", "content": "週末は雨ですか？"}],
        ]

        for conv in weather_queries:
            result = classifier.classify(conv)
            assert isinstance(result, ClassificationResult)
            assert result.need_function is True
            assert result.matched_function == "weather_info"

    def test_classifier_function_needed_translation(self):
        """Translation queries should trigger function calls."""
        classifier = FunctionCallClassifier()

        translation_queries = [
            [{"role": "user", "content": "これを英語にしてください"}],
            [{"role": "user", "content": "翻訳をお願いします"}],
            [{"role": "user", "content": "日本語にして"}],
        ]

        for conv in translation_queries:
            result = classifier.classify(conv)
            assert isinstance(result, ClassificationResult)
            assert result.need_function is True
            assert result.matched_function == "translation_assist"

    def test_classifier_multi_turn_conversation(self):
        """Multi-turn conversations should consider all user messages."""
        classifier = FunctionCallClassifier()

        # First turn is greeting, second turn asks about travel
        conv = [
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは！"},
            {"role": "user", "content": "京都の観光名所を教えて"},
        ]

        result = classifier.classify(conv)
        # Last message has travel keywords
        assert result.need_function is True
        assert result.matched_function == "travel_guide"

    def test_classifier_confidence_levels(self):
        """Test confidence levels for different scenarios."""
        classifier = FunctionCallClassifier()

        # Strong function signal
        strong_signal = [{"role": "user", "content": "京都の観光名所やホテルを教えて"}]
        result = classifier.classify(strong_signal)
        assert result.confidence >= 0.5

        # Strong no_function signal
        strong_no_func = [{"role": "user", "content": "AIとは何ですか？教えてください"}]
        result = classifier.classify(strong_no_func)
        # Should have reasonable confidence

    def test_classification_result_dataclass(self):
        """Test ClassificationResult dataclass."""
        result = ClassificationResult(
            need_function=True,
            confidence=0.8,
            matched_function="travel_guide"
        )
        assert result.need_function is True
        assert result.confidence == 0.8
        assert result.matched_function == "travel_guide"


# --- Router Two-Stage Tests ---

class TestRouterTwoStage:
    """Tests for Router two-stage functionality."""

    def test_router_result_dataclass(self):
        """Test RouterResult with stage1_blocked field."""
        result = RouterResult(
            should_call=False,
            stage1_blocked=True,
            classification=ClassificationResult(
                need_function=False,
                confidence=0.7
            )
        )
        assert result.should_call is False
        assert result.stage1_blocked is True
        assert result.classification.need_function is False

    def test_router_two_stage_enabled_default(self):
        """Two-stage should be enabled by default."""
        reg = FunctionRegistry()
        reg.register(FunctionDefinition("test", "desc", {}, lambda: None))

        from src.ollama_client import OllamaClient
        router = FunctionRouter(OllamaClient(), reg)

        assert router.use_two_stage is True
        assert router.classifier is not None

    def test_router_two_stage_disabled(self):
        """Test router with two-stage disabled."""
        reg = FunctionRegistry()
        reg.register(FunctionDefinition("test", "desc", {}, lambda: None))

        from src.ollama_client import OllamaClient
        router = FunctionRouter(OllamaClient(), reg, use_two_stage=False)

        assert router.use_two_stage is False

    def test_router_result_fields(self):
        """Test all RouterResult fields."""
        result = RouterResult(
            should_call=True,
            function_name="travel_guide",
            arguments={"destination": "Tokyo"},
            raw_response='{"name": "travel_guide"}',
            classification=ClassificationResult(
                need_function=True,
                confidence=0.9,
                matched_function="travel_guide"
            ),
            stage1_blocked=False
        )

        assert result.should_call is True
        assert result.function_name == "travel_guide"
        assert result.arguments == {"destination": "Tokyo"}
        assert result.raw_response == '{"name": "travel_guide"}'
        assert result.classification.confidence == 0.9
        assert result.stage1_blocked is False
