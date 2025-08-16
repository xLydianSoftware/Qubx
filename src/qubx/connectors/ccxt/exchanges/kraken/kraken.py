import ccxt.pro as cxp

from ..base import CcxtFuturePatchMixin


class CustomKrakenFutures(CcxtFuturePatchMixin, cxp.krakenfutures):
    def describe(self):
        # we disable watchBidsAsks because it does not work well
        return self.deep_extend(
            super().describe(),
            {
                "has": {
                    "watchBidsAsks": False,
                }
            },
        )
