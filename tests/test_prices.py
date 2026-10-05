"""One numeric convention for money: integers in, decimals out.

The bug that forced this: floats from the market data fetcher met Decimals
from PostgreSQL's numeric columns in the same expression, and position
monitoring stopped for a 13-position book —

    Error monitoring AMRX: unsupported operand type(s) for -: 'float' and
    'decimal.Decimal'

— so no stop-loss and no profit target was evaluated for any of them.

Tick size follows price: below $52 three decimals, at or above two. CDXS at
$1.34 moves 0.75% per penny, which is where the third decimal earns its place.
"""

from decimal import Decimal

import pytest

from falcon_core import prices


# ── the mixed-type failure ──────────────────────────────────────────────

def test_a_decimal_and_a_float_can_be_compared():
    """The exact expression that raised: DB Decimal minus fetched float."""
    entry = Decimal("17.35")      # as PostgreSQL returns it
    current = 20.48               # as the fetcher returns it
    assert prices.change(entry, current) == Decimal("3.13")
    assert prices.change_pct(entry, current) == pytest.approx(18.04, abs=0.01)


def test_pnl_across_the_two_types():
    assert prices.pnl(Decimal("17.35"), 20.48, 164) == Decimal("513.32")


def test_numpy_scalars_are_accepted():
    np = pytest.importorskip("numpy")
    assert prices.to_mills(np.float64(9.46)) == 9460
    assert prices.notional(np.float64(9.46), np.int64(158)) == Decimal("1494.68")


# ── float noise does not get in ─────────────────────────────────────────

def test_a_float_does_not_bring_its_binary_tail():
    """Decimal(0.1) is 0.1000000000000000055511151231257827; str() is not."""
    assert prices.as_decimal(0.1) == Decimal("0.1")


def test_a_hundred_additions_do_not_drift():
    total = sum(prices.to_mills(0.001) for _ in range(100))
    assert prices.from_mills(total) == Decimal("0.100")


def test_two_prices_that_render_the_same_are_the_same_integer():
    assert prices.to_mills(1.3449999) == prices.to_mills(Decimal("1.345"))


# ── tick size follows the price ─────────────────────────────────────────

@pytest.mark.parametrize("price,expected", [
    (1.3449, Decimal("1.345")),
    (9.4567, Decimal("9.457")),
    (51.9999, Decimal("52.000")),   # below the threshold, so three decimals
    (52.0, Decimal("52.00")),
    (69.2449, Decimal("69.24")),
    (760.9876, Decimal("760.99")),
])
def test_quantization_at_each_size(price, expected):
    assert prices.quantize(price) == expected


def test_the_threshold_is_the_operators_fifty_two_dollars():
    assert prices.FINE_TICK_BELOW == Decimal("52")
    assert prices.tick(51.99) == Decimal("0.001")
    assert prices.tick(52.00) == Decimal("0.01")


def test_a_penny_stock_keeps_its_third_decimal():
    """CDXS at $1.345: rounding to cents would move it 0.37%."""
    assert prices.to_mills(1.345) == 1345
    assert prices.from_mills(1345) == Decimal("1.345")


def test_half_rounds_up_like_money():
    assert prices.quantize(Decimal("1.3455")) == Decimal("1.346")
    assert prices.quantize(Decimal("69.245")) == Decimal("69.25")


# ── mills round-trip ────────────────────────────────────────────────────

@pytest.mark.parametrize("price", ["0.001", "1.345", "9.46", "33.87", "51.999",
                                   "52.00", "69.24", "760.99"])
def test_round_trip_through_mills(price):
    assert prices.from_mills(prices.to_mills(price)) == Decimal(price)


def test_mills_are_integers():
    assert isinstance(prices.to_mills(33.87), int)
    assert prices.to_mills(33.87) == 33870


# ── totals ──────────────────────────────────────────────────────────────

def test_notional_is_to_the_cent():
    assert prices.notional(32.86, 45) == Decimal("1478.70")
    assert prices.notional(69.24, 21) == Decimal("1454.04")


def test_notional_of_a_sub_penny_price():
    """1588 shares of CDXS at $1.345 — the cents all come from the mills."""
    assert prices.notional(Decimal("1.345"), 1588) == Decimal("2135.86")


def test_notional_accepts_a_fractional_quantity():
    assert prices.notional(10.0, Decimal("2.5")) == Decimal("25.00")


def test_pnl_of_a_loss_is_negative():
    assert prices.pnl(3.03, 2.54, 808) == Decimal("-395.92")


# ── percentages ─────────────────────────────────────────────────────────

def test_change_pct_matches_the_dashboard():
    assert prices.change_pct(14.27, 18.05) == pytest.approx(26.49, abs=0.01)


def test_change_pct_of_a_zero_base_is_none_not_an_error():
    """A position with no entry price must not crash the monitor."""
    assert prices.change_pct(0, 10.0) is None


def test_change_is_signed():
    assert prices.change(12.28, 10.93) == Decimal("-1.350")


# ── coercion ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (Decimal("1.25"), Decimal("1.25")),
    (5, Decimal("5")),
    ("1.25", Decimal("1.25")),
    ("  1.25  ", Decimal("1.25")),
    ("", Decimal("0")),
    (1.25, Decimal("1.25")),
])
def test_as_decimal(value, expected):
    assert prices.as_decimal(value) == expected


def test_to_float_is_quantized():
    assert prices.to_float(1.3449999) == 1.345
    assert prices.to_float(69.2449) == 69.24
