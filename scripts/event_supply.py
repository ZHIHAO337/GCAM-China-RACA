from __future__ import annotations

from collections.abc import Mapping
import math


def event_transfer_support(row, multiplier: float) -> float:
    """Use the same pre-event support portfolio as the capacity module."""
    def value(name):
        return float(row[name] if isinstance(row, Mapping) else getattr(row, name))

    loss = value('trade_loss')
    available = value('available_transfer_support_gw')
    if not all(math.isfinite(x) for x in (loss, available, multiplier)):
        raise ValueError('Non-finite transfer input')
    if not 0 <= loss < 1 or available < 0 or multiplier < 0 or loss * multiplier > 1:
        raise ValueError('Invalid transfer availability or event multiplier')
    nominal = available / (1.0 - loss)
    return nominal * (1.0 - loss * multiplier)
