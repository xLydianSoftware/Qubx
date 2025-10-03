"""
Base classes and mixins for CCXT exchange implementations.
"""

import asyncio


class CcxtFuturePatchMixin:
    """
    Mixin class that patches CCXT Future.race to prevent InvalidStateError race conditions.
    
    This fix should be applied to all CCXT exchange classes to prevent race conditions
    that can occur when multiple futures complete simultaneously and try to set the
    result on an already-done future.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Patch the Future.race method to prevent InvalidStateError
        self._patch_future_race()
    
    def _patch_future_race(self):
        """Patch CCXT Future.race to prevent InvalidStateError race condition."""
        from ccxt.async_support.base.ws.future import Future
        
        @classmethod
        def patched_race(cls, futures):
            future = Future()
            coro = asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
            task = asyncio.create_task(coro)
            
            def callback(done):
                complete, _ = done.result()
                # Check for exceptions
                exceptions = []
                cancelled = False
                for f in complete:
                    if f.cancelled():
                        cancelled = True
                    else:
                        err = f.exception()
                        if err:
                            exceptions.append(err)
                
                # Only set result/exception if future is not already done (prevents InvalidStateError)
                if future.done():
                    return
                
                # If any exceptions return with first exception
                if len(exceptions) > 0:
                    future.set_exception(exceptions[0])
                # Else return first result
                elif cancelled:
                    future.cancel()
                else:
                    first_result = list(complete)[0].result()
                    future.set_result(first_result)
            
            task.add_done_callback(callback)
            return future
        
        # Apply the patch
        Future.race = patched_race