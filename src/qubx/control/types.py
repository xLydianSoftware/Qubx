from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionParam:
    """Describes a single parameter for an action."""

    name: str
    type: str  # "string", "number", "integer", "boolean", "enum", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    choices: list[Any] | None = None  # for enum type
    items_type: str | None = None  # for array type


@dataclass
class ActionDef:
    """Describes a callable action on a running bot."""

    name: str
    description: str
    params: list[ActionParam] = field(default_factory=list)
    category: str = "custom"  # "trading", "universe", "diagnostics", "config", "custom"
    read_only: bool = False
    dangerous: bool = False
    hidden: bool = False


@dataclass
class ActionResult:
    """Result of executing an action."""

    status: str  # "ok", "error"
    data: Any = None
    message: str | None = None
    error: str | None = None
