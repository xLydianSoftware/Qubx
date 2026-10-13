# Venue/account-level exchange names whose instruments belong to another exchange's universe,
# keyed canonical -> venue: BINANCE.PM is a portfolio-margin account that trades BINANCE.UM
# instruments. Used by the data-side mixins to fall back from an instrument's canonical
# exchange to a data provider registered under the venue name.
EXCHANGE_MAPPINGS = {
    "BINANCE.UM": "BINANCE.PM",
}

_VENUE_TO_CANONICAL = {venue: canonical for canonical, venue in EXCHANGE_MAPPINGS.items()}


def canonical_exchange(exchange: str) -> str:
    """Canonical (instrument-universe) exchange for a configured venue name: BINANCE.PM -> BINANCE.UM.

    Account-state keys, instruments and connector identities all use the canonical name;
    the venue name survives only for credentials/settings lookups and venue plumbing.
    Names without a mapping are already canonical and pass through (uppercased).
    """
    exchange = exchange.upper()
    return _VENUE_TO_CANONICAL.get(exchange, exchange)
