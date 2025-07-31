import math
from collections import defaultdict, deque
from typing import Any, Callable, Iterator, List, Optional, Set, TypeAlias

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import DataType, Timestamped
from qubx.data.readers import DataReader, DataTransformer

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

    def put(self, data: dict[str, Iterator[list[Timestamped]]]):
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

    def __add__(self, data: dict[str, Iterator]) -> "IteratedDataStreamsSlicer":
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
        except (StopIteration, IndexError):
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


class CompositeReader(DataReader):
    """
    A data reader that combines data from multiple readers.

    This reader will try to retrieve data from each of the provided readers,
    combine the results in a sorted way, and remove duplicates.

    Args:
        readers: A list of DataReader instances to combine
    """

    def __init__(self, readers: List[DataReader]) -> None:
        """
        Initialize the CompositeReader with a list of readers.

        Args:
            readers: A list of DataReader instances to combine
        """
        self.readers = readers
        # logger.debug(f"Created CompositeReader with {len(readers)} readers")

    def get_names(self, **kwargs) -> List[str]:
        """
        Get a combined list of names from all readers.

        Returns:
            A list of unique names from all readers
        """
        names = set()
        for reader in self.readers:
            try:
                reader_names = reader.get_names(**kwargs)
                names.update(reader_names)
            except NotImplementedError:
                logger.debug(f"Reader {reader.__class__.__name__} does not implement get_names")
            except Exception as e:
                logger.warning(f"Error getting names from reader {reader.__class__.__name__}: {e}")

        return sorted(list(names))

    def read(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize=0,
        **kwargs,
    ) -> Iterator | list | None:
        """
        Read data from all readers, combine, sort, and remove duplicates.

        Args:
            data_id: The data identifier to read
            start: The start time for the data
            stop: The stop time for the data
            transform: A DataTransformer to apply to the data
            chunksize: The chunk size for reading data
            **kwargs: Additional arguments to pass to the readers

        Returns:
            Combined data from all readers
        """
        # If chunksize is 0, we can load all data at once
        if chunksize == 0:
            return self._read_all_at_once(data_id, start, stop, transform, **kwargs)

        # Otherwise, we need to create a joint iterator
        return self._read_chunked(data_id, start, stop, transform, chunksize, **kwargs)

    def _read_all_at_once(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        **kwargs,
    ) -> list | None:
        """
        Read all data at once from all readers, combine, sort, and remove duplicates.

        This method is used when chunksize is 0.
        """
        combined_data = []
        _basic_transform = DataTransformer()
        _column_names = []

        # Try to read from each reader
        for reader in self.readers:
            try:
                reader_data = reader.read(
                    data_id=data_id,
                    start=start,
                    stop=stop,
                    transform=_basic_transform,  # Use empty transformer to get raw data
                    chunksize=0,  # Get all data at once
                    **kwargs,
                )
                if not _column_names:
                    _column_names = _basic_transform._column_names
                elif len(_basic_transform._column_names) < len(_column_names):
                    # Take the shorter column names
                    _column_names = _basic_transform._column_names

                # Convert iterator to list if needed
                if isinstance(reader_data, Iterator):
                    reader_data = list(reader_data)

                if reader_data:
                    combined_data.extend(reader_data)
                    # logger.debug(f"Got {len(reader_data)} records from {reader.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Error reading data from {reader.__class__.__name__}: {e}")

        if not combined_data:
            logger.warning(f"No data found for {data_id} in any reader")
            return None

        # Sort the combined data by timestamp
        # Assuming the first element of each record is the timestamp
        combined_data.sort(key=lambda x: x[0])

        # Remove duplicates
        # Assuming the first element of each record is the timestamp
        deduplicated_data = []
        prev_timestamp = None

        for record in combined_data:
            current_timestamp = record[0]

            # Skip if this timestamp is the same as the previous one
            if prev_timestamp is not None and current_timestamp == prev_timestamp:
                continue

            if len(record) > len(_column_names):
                record = record[: len(_column_names)]

            deduplicated_data.append(record)
            prev_timestamp = current_timestamp

        # logger.debug(f"Combined {len(combined_data)} records, deduplicated to {len(deduplicated_data)} records")

        transform.start_transform(data_id, _column_names, start=start, stop=stop)
        transform.process_data(deduplicated_data)
        return transform.collect()

    def _read_chunked(
        self,
        data_id: str,
        start: str | None = None,
        stop: str | None = None,
        transform: DataTransformer = DataTransformer(),
        chunksize: int = 1000,
        **kwargs,
    ) -> Iterator:
        """
        Read data in chunks from all readers, combine, sort, and remove duplicates.

        This method creates a joint iterator that combines data from each reader's iterator,
        processes chunks incrementally, and maintains deduplication across chunks.
        """
        # logger.debug(f"Starting chunked read for {data_id} with chunk size {chunksize}")

        # Create iterators for each reader
        reader_iterators = []
        _basic_transforms = []

        for reader in self.readers:
            try:
                reader_data = reader.read(
                    data_id=data_id,
                    start=start,
                    stop=stop,
                    transform=(_transform := DataTransformer()),
                    chunksize=chunksize,
                    **kwargs,
                )

                # Only add iterators that return data
                if reader_data:
                    reader_iterators.append(reader_data)
                    _basic_transforms.append(_transform)

            except Exception as e:
                logger.warning(f"Error creating iterator from {reader.__class__.__name__}: {e}")

        if not reader_iterators:
            logger.warning(f"No data found for {data_id} in any reader")
            return iter([])

        slicer = IteratedDataStreamsSlicer(lambda x: x[0].timestamp())  # type: ignore
        slicer.put({f"reader_{idx}": it for idx, it in enumerate(reader_iterators)})

        # - after put each transform should have been initialized so we can get the least number of columns
        _column_names = _basic_transforms[0]._column_names
        for _basic_transform in _basic_transforms[1:]:
            if len(_basic_transform._column_names) < len(_column_names):
                _column_names = _basic_transform._column_names

        def joint_chunked_iterator():
            _buffer = []
            _prev_ts = None
            for _, _ts, _data in slicer:
                if _prev_ts is not None and _ts == _prev_ts:
                    continue

                # TODO: cut out common columns
                if len(_data) > len(_column_names):
                    _data = _data[: len(_column_names)]

                _buffer.append(_data)
                _prev_ts = _ts
                if len(_buffer) >= chunksize:
                    transform.start_transform(data_id, _column_names, start=start, stop=stop)
                    transform.process_data(_buffer)
                    yield transform.collect()
                    _buffer = []
            if _buffer:
                transform.start_transform(data_id, _column_names, start=start, stop=stop)
                transform.process_data(_buffer)
                yield transform.collect()

        return joint_chunked_iterator()

    def get_aux_data_ids(self) -> Set[str]:
        """
        Get a combined set of auxiliary data IDs from all readers.

        Returns:
            A set of unique auxiliary data IDs from all readers
        """
        aux_data_ids = set()
        for reader in self.readers:
            try:
                reader_aux_data_ids = reader.get_aux_data_ids()
                aux_data_ids.update(reader_aux_data_ids)
            except Exception as e:
                logger.warning(f"Error getting aux data IDs from reader {reader.__class__.__name__}: {e}")

        return aux_data_ids

    def get_aux_data(self, data_id: str, merge_strategy: str = "concat", tolerance: str = "1min", keep: str = "last", **kwargs) -> Any:
        """
        Get auxiliary data from all readers that have it and merge the results.

        Args:
            data_id: The auxiliary data ID to get
            merge_strategy: How to merge data from multiple readers:
                - "first": Return data from first reader (backward compatibility)
                - "concat": Concatenate DataFrames/Series with tolerance-based deduplication (default)
                - "outer": Outer join on index (for tabular data)
                - "inner": Inner join on index (intersection of indices)
            tolerance: Time tolerance for deduplication (default "1min"). Only applies to concat strategy.
            keep: Which record to keep when duplicates found within tolerance ("last" or "first")
            **kwargs: Additional arguments to pass to the readers

        Returns:
            Merged auxiliary data from all readers that have it

        Raises:
            ValueError: If no reader has the requested auxiliary data
        """
        collected_data = []
        reader_names = []
        
        for i, reader in enumerate(self.readers):
            try:
                data = reader.get_aux_data(data_id, **kwargs)
                collected_data.append(data)
                reader_names.append(f"{reader.__class__.__name__}_{i}")
                logger.debug(f"Got aux data '{data_id}' from reader {reader.__class__.__name__}")
            except ValueError:
                continue
            except Exception as e:
                logger.warning(f"Error getting aux data '{data_id}' from reader {reader.__class__.__name__}: {e}")

        if not collected_data:
            raise ValueError(f"No reader has auxiliary data for '{data_id}'")

        # If only one reader has the data, return it directly
        if len(collected_data) == 1:
            return collected_data[0]

        # Handle different merge strategies
        return self._merge_aux_data(collected_data, reader_names, merge_strategy, data_id, tolerance, keep)

    def _merge_aux_data(self, data_list: List[Any], reader_names: List[str], merge_strategy: str, data_id: str, tolerance: str = "1min", keep: str = "last") -> Any:
        """
        Merge auxiliary data from multiple readers based on the specified strategy.
        
        Args:
            data_list: List of data objects from different readers
            reader_names: Names of the readers that provided the data
            merge_strategy: Strategy to use for merging
            data_id: ID of the auxiliary data being merged
            tolerance: Time tolerance for deduplication in concat strategy
            keep: Which record to keep when duplicates found ("last" or "first")
            
        Returns:
            Merged data object
        """
        if merge_strategy == "first":
            return data_list[0]
            
        # Check if all data are pandas DataFrames or Series
        all_pandas = all(isinstance(data, (pd.DataFrame, pd.Series)) for data in data_list)
        
        if not all_pandas:
            logger.warning(f"Not all aux data for '{data_id}' are pandas objects. "
                         f"Using 'first' strategy as fallback.")
            return data_list[0]
            
        try:
            if merge_strategy == "concat":
                # Use tolerance-based concatenation for intelligent deduplication
                return self._concat_with_tolerance(data_list, data_id, tolerance, keep)
                    
            elif merge_strategy in ["outer", "inner"]:
                # Join DataFrames/Series on their index
                if all(isinstance(data, pd.DataFrame) for data in data_list):
                    result = data_list[0]
                    for i, data in enumerate(data_list[1:], 1):
                        # Add suffix to avoid column name conflicts
                        data_suffixed = data.add_suffix(f"_{reader_names[i]}")
                        result = result.join(data_suffixed, how=merge_strategy, rsuffix=f"_{reader_names[0]}")
                    return result
                    
                elif all(isinstance(data, pd.Series) for data in data_list):
                    # For Series, create a DataFrame with each series as a column
                    result_dict = {}
                    for i, data in enumerate(data_list):
                        result_dict[reader_names[i]] = data
                    return pd.DataFrame(result_dict).dropna(how='all' if merge_strategy == 'outer' else 'any')
                    
            else:
                logger.warning(f"Unknown merge strategy '{merge_strategy}' for aux data '{data_id}'. "
                             f"Using 'first' strategy as fallback.")
                return data_list[0]
                
        except Exception as e:
            logger.error(f"Error merging aux data '{data_id}' with strategy '{merge_strategy}': {e}")
            logger.info(f"Falling back to 'first' strategy for aux data '{data_id}'")
            return data_list[0]
            
        # Fallback
        return data_list[0]

    def _concat_with_tolerance(self, data_list: List[Any], data_id: str, tolerance: str = "1min", keep: str = "last") -> Any:
        """
        Concatenate pandas data with tolerance-based deduplication.
        
        Args:
            data_list: List of pandas DataFrames or Series
            data_id: ID of the data being merged (for logging)
            tolerance: Time tolerance for considering records as duplicates
            keep: Which record to keep when duplicates found ("last" or "first")
            
        Returns:
            Concatenated and deduplicated pandas object
        """
        # Handle different pandas object types
        if all(isinstance(data, pd.DataFrame) for data in data_list):
            return self._concat_dataframes_with_tolerance(data_list, data_id, tolerance, keep)
        elif all(isinstance(data, pd.Series) for data in data_list):
            return self._concat_series_with_tolerance(data_list, data_id, tolerance, keep)
        else:
            logger.warning(f"Mixed pandas object types for '{data_id}'. Using simple concatenation.")
            result = pd.concat(data_list, axis=0, ignore_index=False, sort=True)
            return result.sort_index()

    def _concat_dataframes_with_tolerance(self, data_list: List[pd.DataFrame], data_id: str, tolerance: str = "1min", keep: str = "last") -> pd.DataFrame:
        """
        Concatenate DataFrames with per-symbol tolerance-based deduplication.
        """
        # Simple concatenation first
        result = pd.concat(data_list, axis=0, ignore_index=False, sort=True)
        
        if len(result) == 0:
            return result
            
        # Check if this is multi-index data (timestamp, symbol)
        if isinstance(result.index, pd.MultiIndex) and len(result.index.names) >= 2:
            # Assume first level is timestamp, second is symbol
            timestamp_level = 0
            symbol_level = 1
            
            return self._deduplicate_multiindex_with_tolerance(result, timestamp_level, symbol_level, tolerance, keep, data_id)
        else:
            # Single index - assume it's timestamp
            return self._deduplicate_single_index_with_tolerance(result, tolerance, keep, data_id)

    def _concat_series_with_tolerance(self, data_list: List[pd.Series], data_id: str, tolerance: str = "1min", keep: str = "last") -> pd.Series:
        """
        Concatenate Series with tolerance-based deduplication.
        """
        result = pd.concat(data_list, axis=0, ignore_index=False)
        
        if len(result) == 0:
            return result
            
        return self._deduplicate_single_index_with_tolerance(result, tolerance, keep, data_id)

    def _deduplicate_multiindex_with_tolerance(self, df: pd.DataFrame, timestamp_level: int, symbol_level: int, tolerance: str, keep: str, data_id: str) -> pd.DataFrame:
        """
        Deduplicate MultiIndex DataFrame with tolerance, processing each symbol separately.
        """
        tolerance_delta = pd.Timedelta(tolerance)
        deduplicated_parts = []
        
        # Group by symbol and process each separately
        symbols = df.index.get_level_values(symbol_level).unique()
        
        for symbol in symbols:
            # Extract data for this symbol
            symbol_data = df[df.index.get_level_values(symbol_level) == symbol].copy()
            
            if len(symbol_data) <= 1:
                deduplicated_parts.append(symbol_data)
                continue
                
            # Get timestamps for this symbol
            timestamps = symbol_data.index.get_level_values(timestamp_level)
            
            # Find groups of timestamps within tolerance
            dedupe_mask = self._find_tolerance_groups(timestamps, tolerance_delta, keep)
            
            # Keep only the selected records - convert to numpy boolean array for iloc
            deduplicated_symbol_data = symbol_data.iloc[dedupe_mask.values]
            deduplicated_parts.append(deduplicated_symbol_data)
        
        # Combine all symbols back together
        if deduplicated_parts:
            final_result = pd.concat(deduplicated_parts, axis=0)
            removed_count = len(df) - len(final_result)
            if removed_count > 0:
                logger.debug(f"Removed {removed_count} near-duplicate records (tolerance={tolerance}) for '{data_id}'")
            return final_result.sort_index()
        else:
            return pd.DataFrame(columns=df.columns)

    def _deduplicate_single_index_with_tolerance(self, data: pd.DataFrame | pd.Series, tolerance: str, keep: str, data_id: str) -> pd.DataFrame | pd.Series:
        """
        Deduplicate single-index pandas object with tolerance.
        """
        if len(data) <= 1:
            return data
            
        tolerance_delta = pd.Timedelta(tolerance)
        timestamps = data.index
        
        # Find groups of timestamps within tolerance
        dedupe_mask = self._find_tolerance_groups(timestamps, tolerance_delta, keep)
        
        # Keep only the selected records - convert to numpy boolean array for iloc
        result = data.iloc[dedupe_mask.values]
        removed_count = len(data) - len(result)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} near-duplicate records (tolerance={tolerance}) for '{data_id}'")
            
        return result.sort_index()

    def _find_tolerance_groups(self, timestamps: pd.Index, tolerance_delta: pd.Timedelta, keep: str) -> pd.Series:
        """
        Find which records to keep when grouping timestamps by tolerance.
        
        Args:
            timestamps: Pandas timestamp index
            tolerance_delta: Tolerance as pd.Timedelta
            keep: "first" or "last"
            
        Returns:
            Boolean mask indicating which records to keep
        """
        if len(timestamps) <= 1:
            return pd.Series([True] * len(timestamps), index=range(len(timestamps)))
            
        # Sort timestamps to process chronologically
        sorted_indices = timestamps.argsort()
        sorted_timestamps = timestamps[sorted_indices]
        
        # Find groups within tolerance
        keep_mask = pd.Series([False] * len(timestamps), index=range(len(timestamps)))
        i = 0
        
        while i < len(sorted_timestamps):
            # Find all timestamps within tolerance of current timestamp
            current_time = sorted_timestamps[i]
            group_end = i
            
            # Extend group to include all timestamps within tolerance
            while (group_end + 1 < len(sorted_timestamps) and 
                   sorted_timestamps[group_end + 1] - current_time <= tolerance_delta):
                group_end += 1
            
            # Choose which record to keep from this group
            if keep == "last":
                chosen_idx = sorted_indices[group_end]
            else:  # keep == "first"
                chosen_idx = sorted_indices[i]
                
            keep_mask.iloc[chosen_idx] = True
            
            # Move to next group
            i = group_end + 1
            
        return keep_mask

    def get_symbols(self, exchange: str, dtype: str) -> list[str]:
        """
        Get a combined list of symbols from all readers.

        Args:
            exchange: The exchange to get symbols for
            dtype: The data type to get symbols for

        Returns:
            A list of unique symbols from all readers
        """
        symbols = set()
        for reader in self.readers:
            try:
                reader_symbols = reader.get_symbols(exchange, dtype)
                symbols.update(reader_symbols)
            except NotImplementedError:
                logger.debug(f"Reader {reader.__class__.__name__} does not implement get_symbols")
            except Exception as e:
                logger.warning(f"Error getting symbols from reader {reader.__class__.__name__}: {e}")

        return sorted(list(symbols))

    def get_time_ranges(self, symbol: str, dtype: DataType) -> tuple[np.datetime64 | None, np.datetime64 | None]:
        """
        Get the combined time range from all readers.

        Args:
            symbol: The symbol to get the time range for
            dtype: The data type to get the time range for

        Returns:
            A tuple of (min_time, max_time) from all readers

        Raises:
            ValueError: If no reader has time range data for the symbol and dtype
        """
        min_times = []
        max_times = []

        for reader in self.readers:
            try:
                min_time, max_time = reader.get_time_ranges(symbol, dtype)
                if min_time is not None:
                    min_times.append(min_time)
                if max_time is not None:
                    max_times.append(max_time)
            except NotImplementedError:
                logger.debug(f"Reader {reader.__class__.__name__} does not implement get_time_ranges")
            except Exception as e:
                logger.warning(f"Error getting time ranges from reader {reader.__class__.__name__}: {e}")

        if not min_times or not max_times:
            return None, None

        return min(min_times), max(max_times)

    def close(self):
        for reader in self.readers:
            if hasattr(reader, "close"):
                reader.close()  # type: ignore
