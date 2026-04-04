from qubx.control.types import ActionDef, ActionParam, ActionResult


class TestActionParam:
    def test_basic_construction(self):
        p = ActionParam(name="symbol", type="string", description="A symbol")
        assert p.name == "symbol"
        assert p.type == "string"
        assert p.required is True
        assert p.default is None
        assert p.choices is None

    def test_optional_with_default(self):
        p = ActionParam(name="price", type="number", description="Limit price", required=False, default=0.0)
        assert p.required is False
        assert p.default == 0.0

    def test_enum_with_choices(self):
        p = ActionParam(name="side", type="enum", description="Order side", choices=["buy", "sell"])
        assert p.choices == ["buy", "sell"]

    def test_array_with_items_type(self):
        p = ActionParam(name="symbols", type="array", description="Symbols", items_type="string")
        assert p.items_type == "string"


class TestActionDef:
    def test_basic_construction(self):
        ad = ActionDef(name="get_positions", description="Get positions")
        assert ad.name == "get_positions"
        assert ad.category == "custom"
        assert ad.read_only is False
        assert ad.dangerous is False
        assert ad.hidden is False
        assert ad.params == []

    def test_full_construction(self):
        ad = ActionDef(
            name="trade",
            description="Place order",
            category="trading",
            read_only=False,
            dangerous=True,
            params=[ActionParam(name="symbol", type="string", description="Symbol")],
        )
        assert ad.category == "trading"
        assert ad.dangerous is True
        assert len(ad.params) == 1


class TestActionResult:
    def test_ok_result(self):
        r = ActionResult(status="ok", data={"value": 42})
        assert r.status == "ok"
        assert r.data == {"value": 42}
        assert r.error is None

    def test_error_result(self):
        r = ActionResult(status="error", error="something failed")
        assert r.status == "error"
        assert r.error == "something failed"
        assert r.data is None
