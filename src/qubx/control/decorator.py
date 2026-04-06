import functools
import inspect
from typing import Any, Callable

from .types import ActionDef, ActionParam, ActionResult

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def action(
    description: str,
    category: str = "custom",
    read_only: bool = False,
    dangerous: bool = False,
    hidden: bool = False,
):
    """Mark a strategy method as a callable control action.

    Parameters are inferred from the method signature (type hints + defaults).
    The method receives `ctx` as first arg (injected, not part of the action schema).
    """

    def decorator(method: Callable) -> Callable:
        action_def = ActionDef(
            name=method.__name__,
            description=description,
            category=category,
            read_only=read_only,
            dangerous=dangerous,
            hidden=hidden,
            params=_infer_params(method),
        )

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)

        setattr(wrapper, "__action__", action_def)
        return wrapper

    return decorator


def _infer_params(method: Callable) -> list[ActionParam]:
    """Infer ActionParam list from method signature, skipping 'self' and 'ctx'."""
    params = []
    sig = inspect.signature(method)
    skip = {"self", "ctx"}

    for name, p in sig.parameters.items():
        if name in skip:
            continue

        annotation = p.annotation if p.annotation != inspect.Parameter.empty else str
        # Handle generic types like list[str]
        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            param_type = "array"
        elif origin is dict:
            param_type = "object"
        else:
            param_type = _TYPE_MAP.get(annotation, "string")

        has_default = p.default != inspect.Parameter.empty

        params.append(
            ActionParam(
                name=name,
                type=param_type,
                description=f"Parameter: {name}",
                required=not has_default,
                default=p.default if has_default else None,
            )
        )

    return params


def state(description: str):
    """Mark a strategy method as a state provider.

    The method is called automatically when building state snapshots.
    It receives `ctx` as first arg (injected). Must be fast and read-only.
    The return value is included under the "custom" key in get_state responses.
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)

        setattr(wrapper, "__state__", description)
        return wrapper

    return decorator


def collect_state(strategy: Any, ctx: Any) -> dict:
    """Scan a strategy instance for @state-decorated methods and collect their values."""
    result = {}
    for attr_name in dir(strategy):
        try:
            attr = getattr(strategy, attr_name, None)
        except Exception:
            continue
        if callable(attr) and hasattr(attr, "__state__"):
            try:
                result[attr.__name__] = attr(ctx)
            except Exception as e:
                result[attr.__name__] = f"error: {e}"
    return result


def collect_state_schema(strategy: Any) -> dict[str, str]:
    """Scan a strategy instance for @state-decorated methods and return their descriptions."""
    schema = {}
    for attr_name in dir(strategy):
        try:
            attr = getattr(strategy, attr_name, None)
        except Exception:
            continue
        if callable(attr) and hasattr(attr, "__state__"):
            schema[attr.__name__] = attr.__state__
    return schema


def collect_actions(strategy: Any) -> list[ActionDef]:
    """Scan a strategy instance for @action-decorated methods."""
    actions = []
    for attr_name in dir(strategy):
        try:
            attr = getattr(strategy, attr_name, None)
        except Exception:
            continue
        if callable(attr) and hasattr(attr, "__action__"):
            actions.append(attr.__action__)
    return actions


def execute_decorated_action(strategy: Any, ctx: Any, name: str, params: dict) -> ActionResult:
    """Find and execute an @action-decorated method by name."""
    for attr_name in dir(strategy):
        try:
            attr = getattr(strategy, attr_name, None)
        except Exception:
            continue
        if callable(attr) and hasattr(attr, "__action__") and attr.__action__.name == name:
            try:
                result = attr(ctx, **params)
                if isinstance(result, ActionResult):
                    return result
                return ActionResult(status="ok", data=result)
            except Exception as e:
                return ActionResult(status="error", error=str(e))
    return ActionResult(status="error", error=f"Unknown action: {name}")
