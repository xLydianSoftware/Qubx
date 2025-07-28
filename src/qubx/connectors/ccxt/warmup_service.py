"""
Warmup service for CCXT data provider.

Handles warmup operations by delegating to appropriate data type handlers.
"""

import asyncio
from collections import defaultdict
from typing import Dict, Set, Tuple

from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument
from qubx.utils.misc import AsyncThreadLoop

from .handlers import DataTypeHandlerFactory


class WarmupService:
    """Service responsible for warming up historical data using data type handlers."""
    
    def __init__(self, 
                 handler_factory: DataTypeHandlerFactory, 
                 channel: CtrlChannel,
                 exchange_id: str,
                 async_loop: AsyncThreadLoop,
                 warmup_timeout: int = 120):
        """
        Initialize warmup service.
        
        Args:
            handler_factory: Factory for creating data type handlers
            channel: Control channel for sending warmup data
            exchange_id: Exchange identifier for logging
            async_loop: Async thread loop for executing warmup tasks
            warmup_timeout: Timeout for warmup operations in seconds
        """
        self._handler_factory = handler_factory
        self._channel = channel
        self._exchange_id = exchange_id
        self._async_loop = async_loop
        self._warmup_timeout = warmup_timeout
    
    def execute_warmup(self, warmups: Dict[Tuple[str, Instrument], str]) -> None:
        """
        Execute warmup for multiple data types and instruments.
        
        Args:
            warmups: Dictionary mapping (subscription_type, instrument) to warmup_period
        """
        if not warmups:
            logger.debug(f"<yellow>{self._exchange_id}</yellow> No warmup data requested")
            return
            
        logger.info(f"<yellow>{self._exchange_id}</yellow> Starting warmup for {len(warmups)} items")
        
        # Group warmups by data type and instrument set for efficient batch processing
        warmup_groups: Dict[str, Dict[str, Set[Instrument]]] = defaultdict(lambda: defaultdict(set))
        
        for (sub_type, instrument), period in warmups.items():
            _sub_type, _params = DataType.from_str(sub_type)
            # Group by data type and period for batch processing
            period_key = f"{period}_{','.join(f'{k}={v}' for k, v in _params.items())}"
            warmup_groups[_sub_type][period_key].add(instrument)
        
        # Create warmup tasks for each data type group
        warmup_tasks = []
        for data_type, period_groups in warmup_groups.items():
            handler = self._handler_factory.get_handler(data_type)
            if handler is None:
                logger.warning(f"<yellow>{self._exchange_id}</yellow> Warmup for {data_type} is not supported")
                continue
                
            # Process each period group within the data type
            for period_key, instruments in period_groups.items():
                # Extract period and params from period_key
                if '_' in period_key:
                    parts = period_key.split('_', 1)
                    period = parts[0]
                    param_str = parts[1] if len(parts) > 1 else ""
                    # Parse params from param_str (format: "key1=val1,key2=val2")
                    params = {}
                    if param_str:
                        for param_pair in param_str.split(','):
                            if '=' in param_pair:
                                k, v = param_pair.split('=', 1)
                                params[k] = v
                else:
                    period = period_key
                    params = {}

                # Create warmup coroutine for this handler and instrument set
                warmup_tasks.append(
                    handler.warmup(
                        instruments=instruments,
                        channel=self._channel,
                        warmup_period=period,
                        **params,
                    )
                )
        
        # Execute all warmup tasks concurrently
        if warmup_tasks:
            async def execute_all_warmups():
                await asyncio.gather(*warmup_tasks)
            
            try:
                self._async_loop.submit(execute_all_warmups()).result(self._warmup_timeout)
                logger.info(f"<yellow>{self._exchange_id}</yellow> Warmup completed successfully")
            except Exception as e:
                logger.error(f"<yellow>{self._exchange_id}</yellow> Warmup failed: {e}")
                raise
        else:
            logger.warning(f"<yellow>{self._exchange_id}</yellow> No valid warmup handlers found")