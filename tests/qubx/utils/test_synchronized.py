import threading
import time

from qubx.utils.misc import synchronized


class Counter:
    def __init__(self) -> None:
        self.inside = 0
        self.max_concurrent = 0

    @synchronized
    def a(self) -> None:
        self._body()

    @synchronized
    def b(self) -> None:
        self._body()

    def _body(self) -> None:
        self.inside += 1
        self.max_concurrent = max(self.max_concurrent, self.inside)
        time.sleep(0.01)
        self.inside -= 1


def test_different_methods_on_same_instance_are_mutually_exclusive():
    c = Counter()
    threads = [threading.Thread(target=c.a) for _ in range(5)]
    threads += [threading.Thread(target=c.b) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.max_concurrent == 1


def test_different_instances_do_not_block_each_other():
    first, second = Counter(), Counter()
    done = threading.Event()

    def hold() -> None:
        first.a()
        done.set()

    t = threading.Thread(target=hold)
    t.start()
    second.a()
    t.join()
    assert done.is_set()


def test_reentrant_on_same_instance():
    class Nested:
        @synchronized
        def outer(self) -> str:
            return self.inner()

        @synchronized
        def inner(self) -> str:
            return "ok"

    assert Nested().outer() == "ok"
