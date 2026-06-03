from qubx.core.basics import CtrlChannel


def test_channel_overflow_drops_oldest():
    ch = CtrlChannel("test", capacity=3)
    ch.start()
    ch.send(("a",))
    ch.send(("b",))
    ch.send(("c",))
    ch.send(("d",))
    received = [ch.receive(timeout=1) for _ in range(3)]
    assert received == [("b",), ("c",), ("d",)]


def test_channel_overflow_counts_drops():
    ch = CtrlChannel("test", capacity=2)
    ch.start()
    for x in ("a", "b", "c", "d"):
        ch.send((x,))
    assert ch.dropped_count == 2


def test_channel_overflow_labels_drops_by_type():
    ch = CtrlChannel("test", capacity=1)
    ch.start()
    ch.send(("a",))
    ch.send(("b",))  # drops the ("a",) tuple
    assert ch.dropped_by_type["tuple"] == 1


def test_channel_default_capacity():
    assert CtrlChannel("test").capacity == 10_000
