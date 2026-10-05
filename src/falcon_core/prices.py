"""One numeric convention for money.

Three numeric types were meeting in the same expressions: floats from the
market data fetcher (via pandas, so sometimes numpy scalars), Decimals from
PostgreSQL's numeric columns, and ints for share counts. On 2026-10-05 that
stopped position monitoring outright for a 13-position book:

    Error monitoring AMRX: unsupported operand type(s) for -: 'float' and
    'decimal.Decimal'

No stop-loss and no profit target was evaluated for any position while that
stood. The float side also carries noise that has no meaning in a price:
(28.63 - 24.37) / 24.37 is not more accurate at seventeen digits.

So prices are handled the way point-of-sale systems handle them: converted to
integers, compared and totalled as integers, and rendered as decimals only for
display and storage. The unit is a **mill** -- one thousandth of a dollar.

Tick size follows the price, as the operator asked: below $52 a price carries
three decimals, at or above it two. US equities quote in pennies above $1.00
(SEC Rule 612), so the third decimal matters exactly where a penny is a large
fraction of the price -- CDXS at $1.34 moves 0.75% per penny.

Storage note: positions.entry_price and friends are numeric(15,2) today, so
PostgreSQL rounds to cents on write regardless of what this module computes.
Widening those columns is a schema change and needs its own approval; until
then, mills survive in memory and in the arithmetic, not in the table.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Union

Number = Union[int, float, Decimal, str]

#: Internal unit: one thousandth of a dollar.
MILLS_PER_DOLLAR = 1000

#: Below this price, quote and compute in thousandths; at or above it, cents.
FINE_TICK_BELOW = Decimal("52")

_FINE_TICK = Decimal("0.001")
_COARSE_TICK = Decimal("0.01")


def as_decimal(value: Number) -> Decimal:
    """A Decimal from anything the system carries, without float noise.

    float goes through str() rather than Decimal(float): Decimal(0.1) is
    0.1000000000000000055511151231257827, which then poisons every total it
    takes part in. numpy scalars are handled by float() first, since
    numpy.float64's str() is a plain number but its repr() is not.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    if isinstance(value, str):
        return Decimal(value.strip() or "0")
    item = getattr(value, "item", None)      # numpy scalar
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):       # pragma: no cover - defensive
            pass
    return Decimal(str(float(value)))


def tick(price: Number) -> Decimal:
    """The smallest meaningful increment for a price of this size."""
    return _FINE_TICK if as_decimal(price) < FINE_TICK_BELOW else _COARSE_TICK


def quantize(price: Number) -> Decimal:
    """A price rounded to its own tick. Half rounds up, as money does."""
    value = as_decimal(price)
    return value.quantize(tick(value), rounding=ROUND_HALF_UP)


def to_mills(price: Number) -> int:
    """A price as whole mills, rounded to its tick first.

    This is the form to compare and total in: two prices that render the same
    are the same integer, and no sum of them drifts.
    """
    return int(quantize(price) * MILLS_PER_DOLLAR)


def from_mills(mills: int) -> Decimal:
    """Mills back to a dollar amount, at the tick its size deserves."""
    return quantize(Decimal(int(mills)) / MILLS_PER_DOLLAR)


def to_float(price: Number) -> float:
    """A quantized price as a float, for formatting and JSON."""
    return float(quantize(price))


def notional(price: Number, quantity: Number) -> Decimal:
    """What `quantity` shares cost, to the cent.

    Quantity can be fractional, so this is Decimal arithmetic on mills rather
    than integer-only, but it starts from the quantized price.
    """
    total = (Decimal(to_mills(price)) * as_decimal(quantity)) / MILLS_PER_DOLLAR
    return total.quantize(_COARSE_TICK, rounding=ROUND_HALF_UP)


def change(from_price: Number, to_price: Number) -> Decimal:
    """Price difference, in dollars, at the finer of the two ticks."""
    difference = Decimal(to_mills(to_price) - to_mills(from_price))
    return (difference / MILLS_PER_DOLLAR).quantize(
        min(tick(from_price), tick(to_price)), rounding=ROUND_HALF_UP)


def change_pct(from_price: Number, to_price: Number) -> Optional[float]:
    """Return as a percentage, computed on integers. None if base is zero.

    A float is returned because callers format and compare it, never total it.
    """
    base = to_mills(from_price)
    if base == 0:
        return None
    return (to_mills(to_price) - base) / base * 100.0


def pnl(entry: Number, exit_price: Number, quantity: Number) -> Decimal:
    """Profit or loss on a round turn, to the cent."""
    difference = Decimal(to_mills(exit_price) - to_mills(entry))
    total = (difference * as_decimal(quantity)) / MILLS_PER_DOLLAR
    return total.quantize(_COARSE_TICK, rounding=ROUND_HALF_UP)
