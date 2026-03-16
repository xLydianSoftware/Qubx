import asyncio
import socket
import ssl
import sys

import aiohttp
import ccxt.pro as cxp

from ..base import CcxtFuturePatchMixin


class OkxFutures(CcxtFuturePatchMixin, cxp.okx):
    """
    OKX perpetual futures exchange class.

    Sets defaultType to 'swap' for perpetual contracts and applies
    the CcxtFuturePatchMixin for race condition fix.
    Forces IPv4 connections because OKX API key IP whitelisting
    typically only covers IPv4 addresses.
    """

    def describe(self):
        return self.deep_extend(
            super().describe(),
            {
                "options": {
                    "defaultType": "swap",
                    "positionSide": "net",
                },
            },
        )

    def open(self):
        if self.asyncio_loop is None:
            if sys.version_info >= (3, 7):
                self.asyncio_loop = asyncio.get_running_loop()
            else:
                self.asyncio_loop = asyncio.get_event_loop()
            self.throttler.loop = self.asyncio_loop  # type: ignore

        if self.ssl_context is None:
            # Create our SSL context object with our CA cert file
            self.ssl_context = ssl.create_default_context(cafile=self.cafile) if self.verify else self.verify
            if self.ssl_context and self.safe_bool(self.options, "include_OS_certificates", False):
                os_default_paths = ssl.get_default_verify_paths()
                if os_default_paths.cafile and os_default_paths.cafile != self.cafile:
                    self.ssl_context.load_verify_locations(cafile=os_default_paths.cafile)

        if self.own_session and self.session is None:
            # Pass this SSL context to aiohttp and create a TCPConnector
            self.tcp_connector = aiohttp.TCPConnector(
                ssl=self.ssl_context, loop=self.asyncio_loop, enable_cleanup_closed=True, family=socket.AF_INET
            )
            self.session = aiohttp.ClientSession(
                loop=self.asyncio_loop, connector=self.tcp_connector, trust_env=self.aiohttp_trust_env
            )
