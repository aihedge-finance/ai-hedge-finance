"""Core type aliases and decimal helpers.

Ported from v1: app/utils.py (d, d_round, d_abs, d_is_close).
The rest of app/utils.py is ported verbatim in ahf/utils/utils.py.
These are extracted here because they are domain primitives used
throughout ahf.domain without pulling in the full utils blob.
"""
from __future__ import annotations

from decimal import ROUND_HALF_DOWN, Decimal
from typing import Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Number = Union[int, float, Decimal, str]

# ---------------------------------------------------------------------------
# Decimal helpers (thin wrappers, identical behaviour to v1)
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")
_ONE = Decimal("1")


def d(value: Number) -> Decimal:
    """Convert a value to Decimal. Equivalent to v1 d()."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def d_round(value: Number, precision: int = 8) -> Decimal:
    """Round to `precision` decimal places using ROUND_HALF_DOWN."""
    quantize_str = Decimal("0." + "0" * precision)
    return d(value).quantize(quantize_str, rounding=ROUND_HALF_DOWN)


def d_abs(value: Number) -> Decimal:
    """Absolute value as Decimal."""
    return abs(d(value))


def d_is_close(a: Number, b: Number, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """True if a ≈ b within tolerances (mirrors math.isclose semantics)."""
    da, db = d(a), d(b)
    diff = abs(da - db)
    return diff <= max(Decimal(str(rel_tol)) * max(abs(da), abs(db)), Decimal(str(abs_tol)))
