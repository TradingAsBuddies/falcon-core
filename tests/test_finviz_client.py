"""Tests for the Finviz Elite CSV client.

No network: the fixtures are real export rows captured on 2026-09-30, headers
and values verbatim.

The regression these exist for: Finviz's column ids moved, the client kept
requesting the old ones, and the export came back with no Volume or Average
Volume column. Both fields defaulted to 0, so every stock looked illiquid, the
screener's liquidity floor dropped all of them, and the dashboard showed no
recommendations for weeks while every run logged "success".
"""

import pytest

from falcon_core.finviz_client import FinvizClient

# A real export: headers exactly as Finviz returns them for the corrected ids.
HEADERS = ("Ticker,Company,Sector,Industry,Market Cap,P/E,"
           "Relative Strength Index (14),Average Volume,Relative Volume,"
           "Price,Change,Volume,Performance (5 Minutes)")
AAPL = "AAPL,Apple Inc,Technology,Consumer Electronics,4900.00B,38.21,54.10,47111.23,0.31,329.62,-0.42%,76806,0.00%"
NVDA = "NVDA,NVIDIA Corp,Technology,Semiconductors,5600.00B,52.10,61.30,124149.86,1.02,228.30,0.85%,541360,0.12%"

# What the client used to ask for: no Volume, no Average Volume column at all.
STALE_HEADERS = "Ticker,Company,Sector,Industry,Market Cap,P/E,Forward P/E,PEG,P/S,Price,Change"
STALE_ROW = "CAG,Conagra Brands,Consumer Defensive,Packaged Foods,6.50B,11.2,10.8,2.1,0.7,13.93,-0.35%"


def _client():
    return FinvizClient(auth_key="unused-in-these-tests")


def _rows(*lines):
    return _client().parse_csv("\n".join(lines))


# ── column ids ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("field,expected", [
    ("price", 65), ("change", 66), ("volume", 67),
    ("avg_volume", 63), ("rel_volume", 64), ("rsi", 59),
])
def test_column_ids_match_finviz_today(field, expected):
    """Verified against the live export on 2026-09-30 by requesting c=1..100."""
    assert FinvizClient.COLUMNS[field] == expected


@pytest.mark.parametrize("field", ["ticker", "price", "volume", "avg_volume"])
def test_liquidity_fields_are_always_requested(field):
    """The liquidity floor needs these; an export without them rejects everything."""
    requested = FinvizClient._columns_for(FinvizClient.DEFAULT_FIELDS).split(",")
    assert str(FinvizClient.COLUMNS[field]) in requested


def test_earnings_columns_extend_the_default_set():
    earnings = FinvizClient._columns_for(FinvizClient.EARNINGS_FIELDS).split(",")
    default = FinvizClient._columns_for(FinvizClient.DEFAULT_FIELDS).split(",")
    assert set(default) <= set(earnings)
    assert str(FinvizClient.COLUMNS["earnings_date"]) in earnings


# ── parsing ─────────────────────────────────────────────────────────────

def test_average_volume_is_converted_from_thousands_to_shares():
    """Finviz sends 47111.23 for Apple's 47.1M ADV; Volume is already shares."""
    row = _rows(HEADERS, AAPL)[0]
    assert row["avg_volume"] == pytest.approx(47_111_230.0)
    assert row["volume"] == 76_806


def test_average_dollar_volume_clears_a_five_million_floor():
    """The number the screener actually gates on, end to end."""
    row = _rows(HEADERS, AAPL)[0]
    assert row["price"] * row["avg_volume"] > 5_000_000


def test_price_and_volume_parse():
    rows = _rows(HEADERS, AAPL, NVDA)
    assert [r["ticker"] for r in rows] == ["AAPL", "NVDA"]
    assert rows[1]["price"] == pytest.approx(228.30)
    assert rows[1]["volume"] == 541_360


@pytest.mark.parametrize("field", ["volume", "avg_volume"])
def test_missing_column_is_none_not_zero(field):
    """The silent failure: absent columns read as 0 and looked like real data.

    None lets the liquidity filter say "missing" instead of reporting a
    fabricated "$0 average dollar volume" rejection.
    """
    row = _rows(STALE_HEADERS, STALE_ROW)[0]
    assert row[field] is None


def test_price_survived_the_stale_column_ids():
    """Why the breakage was invisible: id 65 landed on Price by luck.

    The old id map asked for 8/9/10 and got Forward P/E, PEG and P/S, but it
    also asked for 65 (meant as RSI), which is Price. Prices looked right on
    every screen while volume was silently absent.
    """
    assert _rows(STALE_HEADERS, STALE_ROW)[0]["price"] == pytest.approx(13.93)


@pytest.mark.parametrize("cell", ["", "-", "N/A", "not a number"])
def test_unparseable_cells_are_none(cell):
    row = _rows(HEADERS, AAPL.replace("47111.23", cell))[0]
    assert row["avg_volume"] is None


def test_price_keeps_thousands_separators_out_of_the_way():
    row = _rows(HEADERS, AAPL.replace("329.62", '"1,329.62"'))[0]
    assert row["price"] == pytest.approx(1329.62)


def test_raw_row_is_preserved():
    assert _rows(HEADERS, AAPL)[0]["_raw"]["Average Volume"] == "47111.23"


# ── sorting tolerates missing fields ────────────────────────────────────

def test_sorting_does_not_crash_when_volume_is_missing(monkeypatch):
    """None is not orderable against a float; this sort used to raise TypeError."""
    client = _client()
    monkeypatch.setattr(client, "fetch_csv",
                        lambda **kw: "\n".join([STALE_HEADERS, STALE_ROW]))
    stocks = client.get_stocks(filters="sh_price_o5", sort_by_5min=True)
    assert [s["ticker"] for s in stocks] == ["CAG"]


# ── market capitalisation units ─────────────────────────────────────────

MEGA_HEADERS = "Ticker,Market Cap,Price"
AAPL_MEGA = "AAPL,4859715.75,332.99"
KO_MEGA = "KO,372901.90,86.67"
CDNA_ROW = "CDNA,3613.63,22.45"


def _mega(*lines):
    return _client().parse_csv("\n".join((MEGA_HEADERS,) + lines))


def test_market_cap_is_converted_from_millions_to_dollars():
    """Finviz exports Market Cap in millions with no suffix.

    Read as dollars, every stock looked smaller than $10B: the trader
    classified all 23 of 2026-10-06's candidates as small_cap, and nothing
    could ever be mid or large cap.
    """
    row = _mega(AAPL_MEGA)[0]
    assert row["market_cap_usd"] == pytest.approx(4.85971575e12)


@pytest.mark.parametrize("line,expected", [
    (AAPL_MEGA, 4.85971575e12),   # $4.86T
    (KO_MEGA, 3.729019e11),       # $373B
    (CDNA_ROW, 3.61363e9),        # $3.6B
])
def test_each_size_lands_in_the_right_decade(line, expected):
    assert _mega(line)[0]["market_cap_usd"] == pytest.approx(expected)


def test_the_raw_cell_is_still_available_for_display():
    row = _mega(AAPL_MEGA)[0]
    assert row["market_cap"] == "4859715.75"


@pytest.mark.parametrize("cell", ["", "-", "N/A", "junk"])
def test_an_unusable_market_cap_is_none(cell):
    assert _mega(f"AAPL,{cell},332.99")[0]["market_cap_usd"] is None


def test_a_missing_market_cap_column_is_none():
    row = _client().parse_csv("Ticker,Price\nAAPL,332.99")[0]
    assert row["market_cap_usd"] is None


def test_a_mega_cap_clears_the_large_cap_threshold():
    """100e9 is the router's large_cap threshold; AAPL must clear it."""
    assert _mega(AAPL_MEGA)[0]["market_cap_usd"] > 100e9
