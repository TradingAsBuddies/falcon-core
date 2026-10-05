"""Query parameters must reach the driver as plain Python values.

numpy.float64 is a subclass of float, so psycopg2 accepts it and adapts it the
way it adapts any float: with repr(). numpy 2 changed that repr, so the SQL
received an unquoted literal:

    INSERT INTO orders (...) VALUES ('CX', 'BUY', 158, np.float64(9.46)...
    ERROR: schema "np" does not exist

On 2026-10-05 the orchestrator could not place one order or update one
position's price, while logging "No entry signal" and "No open positions to
monitor". Prices come from pandas, so this affected most writes.
"""

import datetime as dt
from decimal import Decimal

import pytest

from falcon_core.db_manager import DatabaseManager

np = pytest.importorskip("numpy")

normalize = DatabaseManager._normalize_params


def test_numpy_float_becomes_a_plain_float():
    (value,) = normalize((np.float64(9.46),))
    assert type(value) is float
    assert value == pytest.approx(9.46)


def test_the_repr_that_broke_postgres_is_gone():
    """The actual failure: repr(np.float64) carries the constructor call."""
    assert "np.float64" in repr(np.float64(9.46))
    (value,) = normalize((np.float64(9.46),))
    assert "np.float64" not in repr(value)


@pytest.mark.parametrize("value,expected_type", [
    (np.float32(1.5), float),
    (np.float64(1.5), float),
    (np.int64(7), int),
    (np.int32(7), int),
    (np.bool_(True), bool),
])
def test_every_numpy_scalar_is_converted(value, expected_type):
    (converted,) = normalize((value,))
    assert type(converted) is expected_type


def test_a_whole_order_row_is_converted():
    """The row shape that failed: symbol, side, quantity, price, timestamp."""
    row = ("CX", "BUY", np.int64(158), np.float64(9.46), "2026-10-05T14:05:00")
    assert [type(v) for v in normalize(row)] == [str, str, int, float, str]


@pytest.mark.parametrize("value", [
    "text", b"bytes", 42, 3.14, True, False, None,
    Decimal("1.25"), dt.datetime(2026, 10, 5), dt.date(2026, 10, 5),
])
def test_ordinary_values_pass_through_unchanged(value):
    (converted,) = normalize((value,))
    assert converted is value or converted == value
    assert type(converted) is type(value)


def test_a_decimal_is_left_for_the_driver():
    """psycopg2 adapts Decimal correctly; converting it would lose precision."""
    (value,) = normalize((Decimal("1.250000000000000001"),))
    assert isinstance(value, Decimal)


def test_numpy_datetime_becomes_a_python_datetime():
    (value,) = normalize((np.datetime64("2026-10-05T14:05:00"),))
    assert isinstance(value, dt.datetime)


def test_lists_and_dicts_are_handled():
    assert [type(v) for v in normalize([np.float64(1.0), "x"])] == [float, str]
    assert type(normalize({"price": np.float64(1.0)})["price"]) is float


@pytest.mark.parametrize("params", [None, (), [], {}])
def test_empty_parameters_are_returned_as_is(params):
    assert normalize(params) == params


def test_a_zero_dimensional_array_is_converted():
    (value,) = normalize((np.array(2.5),))
    assert type(value) is float


def test_a_real_array_is_left_alone():
    """An array parameter is the caller's problem, not silently flattened."""
    array = np.array([1.0, 2.0])
    (value,) = normalize((array,))
    assert value is array
