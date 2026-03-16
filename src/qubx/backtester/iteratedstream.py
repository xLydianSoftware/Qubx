import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeAlias

from qubx.core.basics import Timestamped

SlicerOutData: TypeAlias = tuple[str, int, Timestamped] | tuple


class IteratedDataStreamsSlicer(Iterator[SlicerOutData]):
    """
    This class manages seamless iteration over multiple time-series data streams,
    ensuring that events are processed in the correct chronological order regardless of their source.
    It supports adding / removing new data streams to the slicer on the fly (during the itration).
    """

    _iterators: dict[str, Iterator[list[Timestamped]]]
    _buffers: dict[str, list[Timestamped]]
    _keys: deque[str]
    _iterating: bool

    def __init__(self, time_func: Callable[[Timestamped], Any] = lambda x: x.time):
        self._buffers = defaultdict(list)
        self._iterators = {}
        self._keys = deque()
        self._iterating = False
        self._time_func = time_func

    def put(self, data: Mapping[str, Iterator[list[Timestamped]]]):
        _rebuild = False
        for k, vi in data.items():
            if k not in self._keys:
                self._iterators[k] = vi
                self._buffers[k] = self._load_next_chunk_to_buffer(k)  # do initial chunk fetching
                self._keys.append(k)
                _rebuild = True

        # - rebuild strategy
        if _rebuild and self._iterating:
            self._build_initial_iteration_seq()

    def __add__(self, data: Mapping[str, Iterator]) -> "IteratedDataStreamsSlicer":
        self.put(data)
        return self

    def remove(self, keys: list[str] | str):
        """
        Remove data iterator and associated keys from the queue.
        If the key is not found, it does nothing.
        """
        _keys = keys if isinstance(keys, list) else [keys]
        _rebuild = False
        for i in _keys:
            # Check and remove from each data structure independently
            removed_any = False

            if i in self._buffers:
                self._buffers.pop(i)
                removed_any = True

            if i in self._iterators:
                self._iterators.pop(i)
                removed_any = True

            if i in self._keys:
                self._keys.remove(i)
                removed_any = True

            if removed_any:
                _rebuild = True

        # - rebuild strategy
        if _rebuild and self._iterating:
            self._build_initial_iteration_seq()

    def __iter__(self) -> Iterator:
        self._build_initial_iteration_seq()
        self._iterating = True
        return self

    def _build_initial_iteration_seq(self):
        _init_seq = {k: self._time_func(self._buffers[k][-1]) for k in self._keys if self._buffers[k]}
        _init_seq = dict(sorted(_init_seq.items(), key=lambda item: item[1]))
        self._keys = deque(_init_seq.keys())

    def _load_next_chunk_to_buffer(self, index: str) -> list[Timestamped]:
        try:
            return list(reversed(next(self._iterators[index])))
        except (StopIteration, IndexError, RuntimeError):
            return []

    def _remove_iterator(self, key: str):
        self._buffers.pop(key)
        self._iterators.pop(key)
        self._keys.remove(key)

    def _pop_top(self, k: str) -> Timestamped:
        """
        Removes and returns the most recent timestamped data element from the buffer associated with the given key.
        If the buffer is empty after popping, it attempts to load the next chunk of data into the buffer.
        If no more data is available, the iterator associated with the key is removed.

        Parameters:
            k (str): The key identifying the data stream buffer to pop from.

        Returns:
            Timestamped: The most recent timestamped data element from the buffer.
        """
        if not self._buffers[k]:
            raise StopIteration

        v = (data := self._buffers[k]).pop()
        if not data:
            try:
                data.extend(self._load_next_chunk_to_buffer(k))  # - get next chunk of data
            except StopIteration:
                self._remove_iterator(k)  # - remove iterable data
        return v

    def fetch_before_time(self, key: str, time_ns: int) -> list[Timestamped]:
        """
        Fetches and returns all timestamped data elements from the buffer associated with the given key
        that have a timestamp earlier than the specified time.

        Parameters:
            - key (str): The key identifying the data stream buffer to fetch from.
            - time_ns (int): The timestamp in nanoseconds. All returned elements will have a timestamp less than this value.

        Returns:
            - list[Timestamped]: A list of timestamped data elements that occur before the specified time.
        """
        values = []
        data = self._buffers[key]
        if not data:
            try:
                data.extend(self._load_next_chunk_to_buffer(key))  # - get next chunk of data
            except StopIteration:
                self._remove_iterator(key)
                # Return empty list if no data is available
                return values

        # Check if data is still empty after attempting to load
        if not data:
            return values

        # pull most past elements
        v = data[-1]
        while self._time_func(v) < time_ns:
            values.append(data.pop())
            if not data:
                try:
                    data.extend(self._load_next_chunk_to_buffer(key))  # - get next chunk of data
                except StopIteration:
                    self._remove_iterator(key)
                    break
                # Check if data is still empty after loading attempt
                if not data:
                    break
            v = data[-1]

        return values

    def __next__(self) -> SlicerOutData:
        """
        Advances the iterator to the next available timestamped data element across all data streams.

        Returns:
            - SlicerOutData: A tuple containing the key of the data stream, the timestamp of the data element, and the data element itself.
            - NoDataContinue: If there are no data streams but scheduler has pending events.

        Raises:
            - StopIteration: If there are no more data elements to iterate over and no scheduled events.
        """
        if not self._keys:
            # DON'T set _iterating = False here! We're still iterating, just temporarily out of data
            # Return sentinel indicating no data streams but iteration could continue
            from qubx.backtester.sentinels import NoDataContinue

            return ("", 0, NoDataContinue())

        _min_t = math.inf
        _min_k = self._keys[0]
        for i in self._keys:
            if not self._buffers[i]:
                continue

            _x = self._buffers[i][-1]
            if self._time_func(_x) < _min_t:
                _min_t = self._time_func(_x)
                _min_k = i

        _v = self._pop_top(_min_k)
        return (_min_k, self._time_func(_v), _v)
