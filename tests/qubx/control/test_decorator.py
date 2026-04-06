
from qubx.control.decorator import (
    action,
    collect_actions,
    collect_state,
    collect_state_schema,
    execute_decorated_action,
    state,
)
from qubx.control.types import ActionDef, ActionResult


class SampleStrategy:
    """Fake strategy with decorated actions for testing."""

    @action(description="Get parameters", category="diagnostics", read_only=True)
    def get_params(self, ctx):
        return {"threshold": 0.7}

    @action(description="Set threshold", category="config")
    def set_threshold(self, ctx, value: float):
        if not 0.0 <= value <= 1.0:
            return ActionResult(status="error", error="Must be between 0 and 1")
        return ActionResult(status="ok", data={"old": 0.7, "new": value})

    @action(description="Rebalance", category="trading", dangerous=True)
    def rebalance(self, ctx, aggressive: bool = False):
        return {"rebalanced": True, "aggressive": aggressive}

    @action(description="Multi-param action")
    def multi_params(self, ctx, name: str, count: int, items: list, flag: bool = True):
        return {"name": name, "count": count, "items": items, "flag": flag}

    def not_an_action(self):
        pass


class TestActionDecorator:
    def test_decorator_stores_action_def(self):
        s = SampleStrategy()
        assert hasattr(s.get_params, "__action__")
        ad = s.get_params.__action__
        assert isinstance(ad, ActionDef)
        assert ad.name == "get_params"
        assert ad.description == "Get parameters"
        assert ad.category == "diagnostics"
        assert ad.read_only is True

    def test_decorator_infers_float_param(self):
        s = SampleStrategy()
        ad = s.set_threshold.__action__
        assert len(ad.params) == 1
        p = ad.params[0]
        assert p.name == "value"
        assert p.type == "number"
        assert p.required is True

    def test_decorator_infers_bool_default(self):
        s = SampleStrategy()
        ad = s.rebalance.__action__
        assert ad.dangerous is True
        assert len(ad.params) == 1
        p = ad.params[0]
        assert p.name == "aggressive"
        assert p.type == "boolean"
        assert p.required is False
        assert p.default is False

    def test_decorator_skips_self_and_ctx(self):
        s = SampleStrategy()
        ad = s.set_threshold.__action__
        param_names = [p.name for p in ad.params]
        assert "self" not in param_names
        assert "ctx" not in param_names

    def test_decorator_infers_multiple_params(self):
        s = SampleStrategy()
        ad = s.multi_params.__action__
        assert len(ad.params) == 4
        types = {p.name: p.type for p in ad.params}
        assert types["name"] == "string"
        assert types["count"] == "integer"
        assert types["items"] == "array"
        assert types["flag"] == "boolean"

    def test_decorated_method_still_callable(self):
        s = SampleStrategy()
        result = s.get_params(ctx=None)
        assert result == {"threshold": 0.7}


class TestCollectActions:
    def test_finds_all_decorated_methods(self):
        s = SampleStrategy()
        actions = collect_actions(s)
        names = {a.name for a in actions}
        assert "get_params" in names
        assert "set_threshold" in names
        assert "rebalance" in names
        assert "multi_params" in names
        assert "not_an_action" not in names

    def test_returns_action_defs(self):
        s = SampleStrategy()
        actions = collect_actions(s)
        for a in actions:
            assert isinstance(a, ActionDef)


class TestExecuteDecoratedAction:
    def test_execute_returns_dict_as_action_result(self):
        s = SampleStrategy()
        result = execute_decorated_action(s, ctx=None, name="get_params", params={})
        assert result.status == "ok"
        assert result.data == {"threshold": 0.7}

    def test_execute_returns_action_result_directly(self):
        s = SampleStrategy()
        result = execute_decorated_action(s, ctx=None, name="set_threshold", params={"value": 0.8})
        assert result.status == "ok"
        assert result.data["new"] == 0.8

    def test_execute_returns_error_from_action(self):
        s = SampleStrategy()
        result = execute_decorated_action(s, ctx=None, name="set_threshold", params={"value": 5.0})
        assert result.status == "error"
        assert result.error is not None and "between 0 and 1" in result.error

    def test_execute_passes_kwargs(self):
        s = SampleStrategy()
        result = execute_decorated_action(s, ctx=None, name="rebalance", params={"aggressive": True})
        assert result.status == "ok"
        assert result.data["aggressive"] is True

    def test_execute_uses_default_kwargs(self):
        s = SampleStrategy()
        result = execute_decorated_action(s, ctx=None, name="rebalance", params={})
        assert result.status == "ok"
        assert result.data["aggressive"] is False

    def test_execute_unknown_action(self):
        s = SampleStrategy()
        result = execute_decorated_action(s, ctx=None, name="nonexistent", params={})
        assert result.status == "error"
        assert result.error is not None and "Unknown action" in result.error

    def test_execute_handles_exception(self):
        class BrokenStrategy:
            @action(description="Explodes")
            def boom(self, ctx):
                raise ValueError("kaboom")

        s = BrokenStrategy()
        result = execute_decorated_action(s, ctx=None, name="boom", params={})
        assert result.status == "error"
        assert result.error is not None and "kaboom" in result.error


class TestStateDecorator:
    def test_decorator_marks_method(self):
        class S:
            @state(description="Current regime")
            def regime(self, ctx):
                return "trending"

        s = S()
        assert hasattr(s.regime, "__state__")
        assert s.regime.__state__ == "Current regime"

    def test_collect_state(self):
        class S:
            @state(description="Regime")
            def regime(self, ctx):
                return "trending"

            @state(description="Score")
            def score(self, ctx):
                return 42

            def not_state(self):
                pass

        s = S()
        result = collect_state(s, ctx=None)
        assert result["regime"] == "trending"
        assert result["score"] == 42
        assert "not_state" not in result

    def test_collect_state_handles_errors(self):
        class S:
            @state(description="Broken")
            def broken(self, ctx):
                raise ValueError("oops")

        s = S()
        result = collect_state(s, ctx=None)
        assert "error:" in result["broken"]

    def test_state_and_action_coexist(self):
        class S:
            @state(description="Regime")
            def regime(self, ctx):
                return "trending"

            @action(description="Get params", read_only=True)
            def get_params(self, ctx):
                return {"x": 1}

        s = S()
        # State works
        result = collect_state(s, ctx=None)
        assert result["regime"] == "trending"
        assert "get_params" not in result

        # Action works
        actions = collect_actions(s)
        assert any(a.name == "get_params" for a in actions)
        assert not any(a.name == "regime" for a in actions)

    def test_collect_state_schema(self):
        class S:
            @state(description="Current regime")
            def regime(self, ctx):
                return "trending"

            @state(description="Score metric")
            def score(self, ctx):
                return 42

        s = S()
        schema = collect_state_schema(s)
        assert schema["regime"] == "Current regime"
        assert schema["score"] == "Score metric"
        assert len(schema) == 2
