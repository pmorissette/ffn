import datetime

import numpy as np
import pandas as pd
import pytest

import ffn


def _make_tz_prices(tz="America/New_York", periods=120, freq="D", hour=0):
    index = pd.date_range("2024-01-01", periods=periods, freq=freq, tz=tz)
    if hour:
        index = index + pd.Timedelta(hours=hour)
    rng = np.random.default_rng(20261119)
    a = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, len(index))), index=index, name="A")
    b = pd.Series(200.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.008, len(index))), index=index, name="B")
    return a, b


@pytest.mark.parametrize("tz", ["UTC", "America/New_York", "Asia/Tokyo"])
@pytest.mark.parametrize(
    "make_bound",
    [
        lambda iso: iso,
        lambda iso: datetime.date.fromisoformat(iso),
        lambda iso: datetime.datetime.fromisoformat(iso),
        lambda iso: pd.Timestamp(iso),
    ],
    ids=["str", "date", "naive-datetime", "naive-timestamp"],
)
@pytest.mark.parametrize("mode", ["both", "start-only", "end-only"])
def test_set_date_range_accepts_naive_bounds_for_tz_aware_series(tz, make_bound, mode):
    prices, _ = _make_tz_prices(tz=tz)
    original = prices.copy(deep=True)
    stats = ffn.PerformanceStats(prices, rf=0.02, annualization_factor=365)

    start_iso, end_iso = "2024-02-01", "2024-04-15"
    start_arg = None if mode == "end-only" else make_bound(start_iso)
    end_arg = None if mode == "start-only" else make_bound(end_iso)

    stats.set_date_range(start=start_arg, end=end_arg)

    start_ts = prices.index[0] if start_arg is None else pd.Timestamp(start_iso, tz=tz)
    end_ts = prices.index[-1] if end_arg is None else pd.Timestamp(end_iso, tz=tz)
    expected = ffn.PerformanceStats(prices.loc[start_ts:end_ts], rf=0.02, annualization_factor=365)

    assert stats.start == expected.start
    assert stats.end == expected.end
    pd.testing.assert_series_equal(stats.stats, expected.stats)
    pd.testing.assert_series_equal(stats.lookback_returns, expected.lookback_returns)
    pd.testing.assert_frame_equal(stats.return_table, expected.return_table)

    stats.set_date_range()
    full_expected = ffn.PerformanceStats(prices, rf=0.02, annualization_factor=365)
    pd.testing.assert_series_equal(stats.stats, full_expected.stats)
    pd.testing.assert_series_equal(prices, original)


@pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
@pytest.mark.parametrize("mode", ["both", "start-only", "end-only"])
def test_group_stats_set_date_range_accepts_naive_bounds_for_tz_aware_series(tz, mode):
    a, b = _make_tz_prices(tz=tz)
    frame = pd.concat([a, b], axis=1)
    original = frame.copy(deep=True)
    group = ffn.GroupStats(frame, annualization_factor=365)
    group.set_riskfree_rate(0.02)

    start_iso, end_iso = "2024-02-01", "2024-04-15"
    start_arg = None if mode == "end-only" else start_iso
    end_arg = None if mode == "start-only" else datetime.date.fromisoformat(end_iso)

    group.set_date_range(start=start_arg, end=end_arg)

    start_ts = frame.index[0] if start_arg is None else pd.Timestamp(start_iso, tz=tz)
    end_ts = frame.index[-1] if end_arg is None else pd.Timestamp(end_iso, tz=tz)
    expected = ffn.GroupStats(frame.loc[start_ts:end_ts], annualization_factor=365)
    expected.set_riskfree_rate(0.02)

    pd.testing.assert_frame_equal(group.stats, expected.stats)
    pd.testing.assert_frame_equal(group.lookback_returns, expected.lookback_returns)

    group.set_date_range()
    full_expected = ffn.GroupStats(frame, annualization_factor=365)
    full_expected.set_riskfree_rate(0.02)
    pd.testing.assert_frame_equal(group.stats, full_expected.stats)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("grouped", [False, True], ids=["performance-stats", "group-stats"])
def test_set_date_range_converts_other_tz_aware_bounds(grouped):
    # America/New_York spans winter (-05:00) and summer (-04:00) offsets across March DST.
    a, b = _make_tz_prices(tz="America/New_York", periods=150, hour=12)
    rf_dates = pd.date_range("2024-01-01", periods=150, freq="D", tz="UTC")
    rf = pd.Series(100.0 * 1.0001 ** np.arange(len(rf_dates)), index=rf_dates, name="rf")

    if grouped:
        stats = ffn.GroupStats(a, b, annualization_factor=365)
        stats.set_riskfree_rate(rf)
    else:
        stats = ffn.PerformanceStats(a, rf=rf, annualization_factor=365)

    start_utc = "2024-02-15T12:00:00Z"
    end_utc = pd.Timestamp("2024-04-15 12:00:00", tz="UTC")
    start_ny = pd.Timestamp(start_utc).tz_convert("America/New_York")
    end_ny = end_utc.tz_convert("America/New_York")

    # Partial updates mix one supplied UTC bound with one stored America/New_York endpoint.
    stats.set_date_range(start=start_utc)
    expected_start_only = (
        ffn.GroupStats(a.loc[start_ny:], b.loc[start_ny:], annualization_factor=365) if grouped else ffn.PerformanceStats(a.loc[start_ny:], rf=rf, annualization_factor=365)
    )
    if grouped:
        expected_start_only.set_riskfree_rate(rf)
        pd.testing.assert_frame_equal(stats.stats, expected_start_only.stats)
    else:
        pd.testing.assert_series_equal(stats.stats, expected_start_only.stats)

    stats.set_date_range(start=start_utc, end=end_utc)
    expected_both = (
        ffn.GroupStats(a.loc[start_ny:end_ny], b.loc[start_ny:end_ny], annualization_factor=365)
        if grouped
        else ffn.PerformanceStats(a.loc[start_ny:end_ny], rf=rf, annualization_factor=365)
    )
    if grouped:
        expected_both.set_riskfree_rate(rf)
        pd.testing.assert_frame_equal(stats.stats, expected_both.stats)
    else:
        pd.testing.assert_series_equal(stats.stats, expected_both.stats)


@pytest.mark.parametrize("grouped", [False, True], ids=["performance-stats", "group-stats"])
def test_set_date_range_preserves_naive_index_rejection_and_empty_range_state(grouped):
    naive_idx = pd.date_range("2024-01-01", periods=10, freq="D")
    naive_a = pd.Series(np.linspace(100.0, 110.0, 10), index=naive_idx, name="A")
    naive_b = pd.Series(np.linspace(200.0, 210.0, 10), index=naive_idx, name="B")
    naive_stats = ffn.GroupStats(naive_a, naive_b) if grouped else ffn.PerformanceStats(naive_a)
    before_naive = naive_stats.stats.copy()

    with pytest.raises((TypeError, ValueError)):
        naive_stats.set_date_range(start="2024-01-02T00:00:00Z")
    with pytest.raises(TypeError):
        naive_stats.set_date_range(start="2024-01-02T00:00:00Z", end="2024-01-05T00:00:00Z")
    if grouped:
        pd.testing.assert_frame_equal(naive_stats.stats, before_naive)
    else:
        pd.testing.assert_series_equal(naive_stats.stats, before_naive)

    tz_a, tz_b = _make_tz_prices(tz="America/New_York", periods=20)
    tz_stats = ffn.GroupStats(tz_a, tz_b) if grouped else ffn.PerformanceStats(tz_a)
    tz_stats.set_date_range("2024-01-03", "2024-01-12")
    before_tz = tz_stats.stats.copy()
    before_children = dict(tz_stats) if grouped else {}

    with pytest.raises(ValueError, match="no usable data"):
        tz_stats.set_date_range("2024-02-01", "2024-02-10")
    if grouped:
        pd.testing.assert_frame_equal(tz_stats.stats, before_tz)
        for name, child in before_children.items():
            assert tz_stats[name] is child
    else:
        pd.testing.assert_series_equal(tz_stats.stats, before_tz)


def test_group_stats_with_mixed_input_timezones_slices_by_naive_date_strings():
    a_utc, b_base = _make_tz_prices(tz="UTC", periods=40)
    b_ny = b_base.tz_convert("America/New_York")
    group = ffn.GroupStats(a_utc, b_ny, annualization_factor=365)

    group.set_date_range("2024-01-10", "2024-02-05")
    expected_start = pd.Timestamp("2024-01-10", tz="UTC")
    expected_end = pd.Timestamp("2024-02-05", tz="UTC")
    expected = ffn.GroupStats(a_utc.loc[expected_start:expected_end], b_ny.loc[expected_start:expected_end], annualization_factor=365)

    pd.testing.assert_frame_equal(group.stats, expected.stats)
