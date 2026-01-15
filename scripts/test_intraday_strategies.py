#!/usr/bin/env python3
"""
Test intraday strategies with synthetic market data.

This script generates realistic 5-minute intraday data to test:
- ATR Breakout Strategy (5-minute gapper breakouts)
- Market Memory Strategy (Day 2 setups)
- One Candle Rule Strategy (1-minute S/R retest)

Since Polygon.io API key is not configured, we use synthetic data
that mimics real market patterns including:
- Opening gaps
- Morning volatility
- Volume spikes
- Support/resistance levels
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import pytz

from falcon_core.backtesting.strategies.atr_breakout import ATRBreakoutStrategy
from falcon_core.backtesting.strategies.market_memory import MarketMemoryStrategy
from falcon_core.backtesting.strategies.one_candle_rule import OneCandleRuleStrategy


def generate_intraday_data(
    symbol: str,
    trading_date: datetime,
    base_price: float = 50.0,
    gap_pct: float = 0.05,
    volatility: float = 0.02,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """
    Generate synthetic intraday data for a single trading day.

    Args:
        symbol: Stock symbol (for logging)
        trading_date: Date for the data
        base_price: Previous day's close price
        gap_pct: Opening gap percentage (positive = gap up)
        volatility: Intraday volatility factor
        interval_minutes: Bar interval in minutes

    Returns:
        DataFrame with OHLCV data indexed by timestamp
    """
    et = pytz.timezone('America/New_York')

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = et.localize(datetime.combine(trading_date.date(), dt_time(9, 30)))
    market_close = et.localize(datetime.combine(trading_date.date(), dt_time(16, 0)))

    # Generate timestamps
    timestamps = []
    current = market_open
    while current < market_close:
        timestamps.append(current)
        current += timedelta(minutes=interval_minutes)

    n_bars = len(timestamps)

    # Generate price path with realistic patterns
    np.random.seed(int(trading_date.timestamp()) % 2**31)

    # Opening gap
    open_price = base_price * (1 + gap_pct)

    # Price changes with mean reversion and trend
    returns = np.random.normal(0, volatility / np.sqrt(78 / n_bars), n_bars)

    # Add morning volatility (first 30 min higher vol)
    morning_bars = min(6, n_bars)  # First 6 bars (30 min for 5-min data)
    returns[:morning_bars] *= 1.5

    # Add trend component (slight drift up or down)
    trend = np.linspace(0, np.random.uniform(-0.02, 0.02), n_bars)
    returns = returns + trend / n_bars

    # Generate OHLC
    closes = [open_price]
    for r in returns[1:]:
        closes.append(closes[-1] * (1 + r))
    closes = np.array(closes)

    # Generate intrabar volatility for high/low
    intrabar_vol = np.abs(np.random.normal(0, volatility * 0.5, n_bars))
    highs = closes * (1 + intrabar_vol)
    lows = closes * (1 - intrabar_vol)

    # Opens are close to previous close with small gap
    opens = np.roll(closes, 1)
    opens[0] = open_price
    opens[1:] = opens[1:] * (1 + np.random.normal(0, 0.001, n_bars - 1))

    # Ensure OHLC consistency
    for i in range(n_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])

    # Generate volume with morning spike
    base_volume = np.random.uniform(100000, 500000, n_bars)
    volume_multiplier = np.ones(n_bars)
    volume_multiplier[:morning_bars] = np.linspace(3, 1.5, morning_bars)  # Morning spike
    volumes = base_volume * volume_multiplier

    # Add volume spikes on big moves
    big_moves = np.abs(returns) > volatility * 1.5
    volumes[big_moves] *= 2

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes.astype(int),
    }, index=timestamps)

    return df


def generate_day2_data(
    symbol: str,
    day1_date: datetime,
    base_price: float = 50.0,
    day1_move_pct: float = 0.08,  # Big Day 1 move
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """
    Generate 2 days of data for Market Memory Day 2 strategy testing.

    Day 1: Big move (gap + trend)
    Day 2: Retest of Day 1 levels with John Wick confirmation
    """
    et = pytz.timezone('America/New_York')
    day2_date = day1_date + timedelta(days=1)

    # Skip weekends
    while day2_date.weekday() >= 5:
        day2_date += timedelta(days=1)

    np.random.seed(42)

    # Day 1 timestamps
    day1_timestamps = []
    current = et.localize(datetime.combine(day1_date.date(), dt_time(9, 30)))
    while current < et.localize(datetime.combine(day1_date.date(), dt_time(16, 0))):
        day1_timestamps.append(current)
        current += timedelta(minutes=interval_minutes)

    n1 = len(day1_timestamps)

    # Day 1: Strong gap and trend up
    day1_open = base_price * (1 + day1_move_pct * 0.5)  # 50% of move is gap
    day1_trend = np.linspace(0, day1_move_pct * 0.5, n1)  # Rest is trend
    day1_closes = day1_open * (1 + day1_trend + np.random.normal(0, 0.003, n1))
    day1_highs = day1_closes * 1.003
    day1_lows = day1_closes * 0.997
    day1_opens = np.roll(day1_closes, 1)
    day1_opens[0] = day1_open

    day1_df = pd.DataFrame({
        'open': day1_opens, 'high': day1_highs, 'low': day1_lows,
        'close': day1_closes, 'volume': np.random.uniform(100000, 200000, n1).astype(int)
    }, index=day1_timestamps)

    yesterday_high = day1_df['high'].max()
    yesterday_low = day1_df['low'].min()

    # Day 2 timestamps
    day2_timestamps = []
    current = et.localize(datetime.combine(day2_date.date(), dt_time(9, 30)))
    while current < et.localize(datetime.combine(day2_date.date(), dt_time(16, 0))):
        day2_timestamps.append(current)
        current += timedelta(minutes=interval_minutes)

    n2 = len(day2_timestamps)
    day2_data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

    # Day 2: Drift down to yesterday's low, John Wick, then bounce
    price = (yesterday_low + yesterday_high) / 2

    for i in range(n2):
        if i < 6:
            # Drift down towards yesterday's low
            new_price = price * 0.995
            day2_data['open'].append(price)
            day2_data['close'].append(new_price)
            day2_data['high'].append(price + 0.1)
            day2_data['low'].append(new_price - 0.1)
            day2_data['volume'].append(150000)
            price = new_price

        elif i == 6:
            # JOHN WICK CANDLE - test yesterday's low with long lower wick
            low_p = yesterday_low * 0.998
            candle_range = 0.50
            body_size = candle_range * 0.10  # 10% body
            lower_wick = candle_range * 0.80  # 80% lower wick

            open_p = low_p + lower_wick
            close_p = open_p + body_size  # Bullish close
            high_p = low_p + candle_range

            day2_data['open'].append(open_p)
            day2_data['close'].append(close_p)
            day2_data['high'].append(high_p)
            day2_data['low'].append(low_p)
            day2_data['volume'].append(250000)
            price = close_p
        else:
            # Bounce back up
            new_price = price * 1.003
            day2_data['open'].append(price)
            day2_data['close'].append(new_price)
            day2_data['high'].append(new_price + 0.1)
            day2_data['low'].append(price - 0.05)
            day2_data['volume'].append(150000)
            price = new_price

    day2_df = pd.DataFrame(day2_data, index=day2_timestamps)

    # Combine Day 1 and Day 2
    combined = pd.concat([day1_df, day2_df])
    combined['date'] = combined.index.date

    return combined


def generate_sr_breakout_data(
    symbol: str,
    trading_date: datetime,
    base_price: float = 50.0,
    interval_minutes: int = 1,
) -> pd.DataFrame:
    """
    Generate data with clear S/R breakout pattern for One Candle Rule testing.

    Creates:
    1. Initial trading range with multiple touches of resistance (30 bars)
    2. Breakout above resistance with volume (5 bars)
    3. Retest of broken resistance with hammer confirmation (10 bars)
    4. Continuation move
    """
    et = pytz.timezone('America/New_York')
    # Use 2-hour window for focused test
    market_open = et.localize(datetime.combine(trading_date.date(), dt_time(9, 30)))
    market_close = et.localize(datetime.combine(trading_date.date(), dt_time(11, 30)))

    timestamps = []
    current = market_open
    while current < market_close:
        timestamps.append(current)
        current += timedelta(minutes=interval_minutes)

    n_bars = len(timestamps)
    np.random.seed(123)

    # Define resistance level
    resistance = base_price + 1.0  # $51 resistance

    data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    price = base_price

    for i in range(n_bars):
        vol = np.random.uniform(100000, 200000)

        if i < 30:  # Phase 1: Range bound with resistance touches
            if i % 6 == 5:  # Touch resistance every 6 bars
                high_p = resistance + np.random.uniform(-0.05, 0.05)
                close_p = resistance - np.random.uniform(0.1, 0.3)
            else:
                close_p = base_price + np.random.uniform(-0.3, 0.3)
                high_p = close_p + np.random.uniform(0.05, 0.2)

            open_p = price
            low_p = min(open_p, close_p) - np.random.uniform(0.05, 0.15)
            price = close_p

        elif i < 35:  # Phase 2: Breakout
            open_p = price
            close_p = resistance + 0.2 + (i - 30) * 0.1
            high_p = close_p + 0.1
            low_p = open_p - 0.05
            vol *= 2.0  # High volume breakout
            price = close_p

        elif i == 40:  # Phase 3: Hammer at retest
            # Proper hammer: small body at top, long lower wick
            open_p = resistance + 0.10
            low_p = resistance - 0.20  # Wick below resistance
            close_p = resistance + 0.15  # Close above resistance (bullish)
            high_p = close_p + 0.02
            vol *= 1.5
            price = close_p

        elif i < 50:  # Pullback to retest area
            open_p = price
            close_p = price - np.random.uniform(0, 0.1)
            high_p = max(open_p, close_p) + 0.05
            low_p = min(open_p, close_p) - 0.05
            price = close_p

        else:  # Phase 4: Continuation up
            open_p = price
            close_p = price + np.random.uniform(0.02, 0.08)
            high_p = close_p + 0.05
            low_p = open_p - 0.03
            price = close_p

        data['open'].append(open_p)
        data['close'].append(close_p)
        data['high'].append(high_p)
        data['low'].append(low_p)
        data['volume'].append(int(vol))

    df = pd.DataFrame(data, index=timestamps)
    return df


def test_atr_breakout():
    """Test ATR Breakout Strategy with synthetic 5-minute gapper data."""
    print("\n" + "="*60)
    print("Testing ATR Breakout Strategy (5-minute data)")
    print("="*60)

    strategy = ATRBreakoutStrategy()
    print(f"Strategy: {strategy.name}")
    print(f"Recommended interval: {strategy.params.recommended_interval}")

    # Generate data for a gapper
    trading_date = datetime(2024, 1, 15)  # A Monday
    data = generate_intraday_data(
        symbol="GAPPER",
        trading_date=trading_date,
        base_price=15.0,
        gap_pct=0.08,  # 8% gap up
        volatility=0.025,
        interval_minutes=5,
    )

    print(f"\nGenerated {len(data)} bars of 5-minute data")
    print(f"Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    print(f"Opening gap: +{((data['open'].iloc[0] / 15.0) - 1) * 100:.1f}%")

    # Calculate indicators
    indicators = strategy.calculate_indicators(data)
    data = pd.concat([data, pd.DataFrame(indicators)], axis=1)

    # Generate signals
    signals = strategy.generate_signals(data)

    print(f"\nSignals generated: {len(signals)}")
    for signal in signals[:5]:  # Show first 5
        print(f"  {signal.timestamp}: {signal.signal_type.name} @ ${signal.price:.2f}")
        print(f"    Reason: {signal.reason}")

    if len(signals) > 5:
        print(f"  ... and {len(signals) - 5} more signals")

    return len(signals) > 0


def test_market_memory():
    """Test Market Memory Strategy with synthetic Day 2 data."""
    print("\n" + "="*60)
    print("Testing Market Memory Strategy (Day 2 setup)")
    print("="*60)

    strategy = MarketMemoryStrategy()
    print(f"Strategy: {strategy.name}")
    print(f"Recommended interval: {strategy.params.recommended_interval}")
    print(f"Min Day 1 move: {strategy.params.min_day1_move_pct * 100:.0f}%")

    # Generate Day 1 + Day 2 data
    day1_date = datetime(2024, 1, 15)  # Monday
    data = generate_day2_data(
        symbol="MOMENTUM",
        day1_date=day1_date,
        base_price=25.0,
        day1_move_pct=0.12,  # 12% Day 1 move
        interval_minutes=5,
    )

    dates = data['date'].unique()
    print(f"\nGenerated {len(data)} bars across {len(dates)} days")

    for d in dates:
        day_data = data[data['date'] == d]
        print(f"  {d}: {len(day_data)} bars, range ${day_data['low'].min():.2f}-${day_data['high'].max():.2f}")

    # Calculate indicators
    indicators = strategy.calculate_indicators(data)
    data_with_ind = pd.concat([data.drop('date', axis=1), pd.DataFrame(indicators)], axis=1)
    data_with_ind['date'] = data['date'].values

    # Generate signals
    signals = strategy.generate_signals(data_with_ind)

    print(f"\nSignals generated: {len(signals)}")
    for signal in signals[:5]:
        print(f"  {signal.timestamp}: {signal.signal_type.name} @ ${signal.price:.2f}")
        print(f"    Reason: {signal.reason}")
        if signal.metadata:
            if 'yesterday_low' in signal.metadata:
                print(f"    Yesterday's low: ${signal.metadata['yesterday_low']:.2f}")
            if 'yesterday_high' in signal.metadata:
                print(f"    Yesterday's high: ${signal.metadata['yesterday_high']:.2f}")

    return len(signals) > 0


def test_one_candle_rule():
    """Test One Candle Rule Strategy with synthetic S/R breakout data."""
    print("\n" + "="*60)
    print("Testing One Candle Rule Strategy (1-minute data)")
    print("="*60)

    strategy = OneCandleRuleStrategy()
    print(f"Strategy: {strategy.name}")
    print(f"Recommended interval: {strategy.params.recommended_interval}")
    print(f"Trading window: {strategy.params.trade_start_time} - {strategy.params.trade_end_time}")

    # Generate data with S/R breakout pattern
    trading_date = datetime(2024, 1, 15)
    data = generate_sr_breakout_data(
        symbol="BREAKOUT",
        trading_date=trading_date,
        base_price=50.0,
        interval_minutes=1,
    )

    print(f"\nGenerated {len(data)} bars of 1-minute data")
    print(f"Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")

    # Calculate indicators
    indicators = strategy.calculate_indicators(data)
    data = pd.concat([data, pd.DataFrame(indicators)], axis=1)

    # Generate signals
    signals = strategy.generate_signals(data)

    print(f"\nSignals generated: {len(signals)}")
    for signal in signals[:5]:
        print(f"  {signal.timestamp}: {signal.signal_type.name} @ ${signal.price:.2f}")
        print(f"    Reason: {signal.reason}")
        if signal.metadata:
            if 'resistance_level' in signal.metadata:
                print(f"    Resistance level: ${signal.metadata['resistance_level']:.2f}")

    return len(signals) > 0


def main():
    print("="*60)
    print("INTRADAY STRATEGY TESTING")
    print("Using synthetic market data (Polygon.io key not configured)")
    print("="*60)

    results = {}

    # Test each strategy
    results['ATR Breakout'] = test_atr_breakout()
    results['Market Memory'] = test_market_memory()
    results['One Candle Rule'] = test_one_candle_rule()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for strategy, success in results.items():
        status = "PASS - signals generated" if success else "NEEDS REVIEW - no signals"
        print(f"  {strategy}: {status}")

    print("\n" + "-"*60)
    print("NOTE: Testing with synthetic data to verify strategy logic.")
    print("For live testing, set POLYGON_API_KEY environment variable")
    print("to fetch real intraday market data from Polygon.io.")
    print("-"*60)

    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
