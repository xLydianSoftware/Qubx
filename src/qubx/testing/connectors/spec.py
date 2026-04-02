"""YAML test spec parsing for connector verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TestSettings(BaseModel):
    max_parallel: int = 3
    timeout_multiplier: float = 2.0


class TestCaseSpec(BaseModel):
    name: str
    subscription: str
    instruments: list[str]
    duration: str
    warmup: str | None = None
    assertions: list[str | dict[str, Any]]

    def parsed_assertions(self) -> list[tuple[str, dict[str, Any]]]:
        """Normalize assertions to (name, params) tuples."""
        result = []
        for a in self.assertions:
            if isinstance(a, str):
                result.append((a, {}))
            elif isinstance(a, dict):
                for name, params in a.items():
                    if isinstance(params, dict):
                        result.append((name, params))
                    else:
                        result.append((name, {"value": params}))
            else:
                raise ValueError(f"Invalid assertion format: {a}")
        return result


class ConnectorTestSpec(BaseModel):
    connector: str
    exchange: str
    settings: TestSettings = Field(default_factory=TestSettings)
    tests: list[TestCaseSpec]

    @classmethod
    def load(cls, path: Path) -> ConnectorTestSpec:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def filter_tests(self, pattern: str) -> ConnectorTestSpec:
        filtered = [t for t in self.tests if pattern in t.name]
        return self.model_copy(update={"tests": filtered})
