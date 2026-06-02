"""
Scheduling helpers for the account-management redesign.

``_ms_to_cron`` converts a millisecond interval into a cron expression the qubx
scheduler (``BasicScheduler`` / ``SimulatedScheduler``, both ``croniter``-backed) can
register. It targets the sub-minute in-flight stuck-order tick; for intervals above one
minute use ``qubx.utils.time.interval_to_cron``.

The qubx scheduler uses 6-field cron with the seconds field LAST (e.g. ``"* * * * * */2"``
fires every 2 seconds), matching ``interval_to_cron``'s output format.
"""


def _ms_to_cron(ms: int) -> str:
    """
    Convert a millisecond interval to a cron expression.

    Supported domain (the in-flight tick is the only consumer):
      - whole-second intervals from 1s to 59s -> ``"* * * * * */<seconds>"``
      - exactly 60_000 ms (one minute)         -> ``"* * * * *"``

    Raises ``ValueError`` for non-positive, sub-second, fractional-second, or
    multi-minute intervals.
    """
    if ms <= 0:
        raise ValueError(f"interval must be positive, got {ms}ms")
    if ms % 1000 != 0:
        raise ValueError(f"interval must be a whole number of seconds, got {ms}ms")

    seconds = ms // 1000
    if seconds == 60:
        return "* * * * *"
    if 1 <= seconds < 60:
        return f"* * * * * */{seconds}"
    raise ValueError(
        f"_ms_to_cron targets sub-minute intervals; {seconds}s is unsupported "
        "(use qubx.utils.time.interval_to_cron for >1min schedules)"
    )
