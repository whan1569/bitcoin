from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any, Callable

import pandas as pd
import requests


BINANCE_FAPI_BASE = "https://fapi.binance.com"
#DATA_ROOT = Path("data")
DATA_ROOT = Path("E:/data")
REQUEST_TIMEOUT = 15
MAX_RETRIES = 6
BASE_SLEEP = 0.25
LS_LIMIT = 500

VALID_PERIODS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

# limit=500 넘지 않게 여유 두고 window 설정
WINDOWS_BY_PERIOD = {
    "5m": timedelta(days=1),     # 288 rows/day
    "15m": timedelta(days=4),    # 384 rows/4day
    "30m": timedelta(days=8),    # 384 rows/8day
    "1h": timedelta(days=16),    # 384 rows/16day
    "2h": timedelta(days=30),    # 360 rows/30day
    "4h": timedelta(days=30),    # 180 rows/30day
    "6h": timedelta(days=30),    # 120 rows/30day
    "12h": timedelta(days=30),   # 60 rows/30day
    "1d": timedelta(days=30),    # 30 rows/30day
}


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_dt(dt: datetime, period_ms: int) -> datetime:
    dt = ensure_utc(dt)
    ts_ms = dt_to_ms(dt)
    floored_ms = (ts_ms // period_ms) * period_ms
    return ms_to_dt(floored_ms)


def parse_start_dt(value: str | int | datetime, period: str) -> datetime:
    period_ms = VALID_PERIODS[period]

    if isinstance(value, datetime):
        return floor_dt(value, period_ms)

    if isinstance(value, int):
        if value > 10_000_000_000:
            dt = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(value, tz=timezone.utc)
        return floor_dt(dt, period_ms)

    s = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return floor_dt(dt, period_ms)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_ls_day_file(symbol: str, period: str, day: datetime) -> Path:
    return DATA_ROOT / symbol.upper() / "LS" / period / f"{day:%Y-%m-%d}.parquet"


def safe_get_json(url: str, params: dict[str, Any]) -> Any:
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 429:
                wait = min(2 ** attempt, 60)
                print(f"[429] Rate limit hit. sleeping {wait}s ...")
                time.sleep(wait)
                continue

            if resp.status_code in (418, 403):
                wait = min(5 * (attempt + 1), 120)
                print(f"[{resp.status_code}] Temp blocked. sleeping {wait}s ...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            last_error = e
            wait = min(2 ** attempt, 30)
            print(f"[retry {attempt + 1}/{MAX_RETRIES}] {e} -> sleep {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Request failed after retries: {last_error}")


def fetch_top_long_short_account_ratio(
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int = LS_LIMIT,
) -> list[dict[str, Any]]:
    url = f"{BINANCE_FAPI_BASE}/futures/data/topLongShortAccountRatio"
    params = {
        "symbol": symbol.upper(),
        "period": period,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    return safe_get_json(url, params)


def fetch_top_long_short_position_ratio(
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int = LS_LIMIT,
) -> list[dict[str, Any]]:
    url = f"{BINANCE_FAPI_BASE}/futures/data/topLongShortPositionRatio"
    params = {
        "symbol": symbol.upper(),
        "period": period,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    return safe_get_json(url, params)


def fetch_global_long_short_account_ratio(
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int = LS_LIMIT,
) -> list[dict[str, Any]]:
    url = f"{BINANCE_FAPI_BASE}/futures/data/globalLongShortAccountRatio"
    params = {
        "symbol": symbol.upper(),
        "period": period,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    return safe_get_json(url, params)


def fetch_open_interest_hist(
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int = LS_LIMIT,
) -> list[dict[str, Any]]:
    url = f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist"
    params = {
        "symbol": symbol.upper(),
        "period": period,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    return safe_get_json(url, params)


def get_last_saved_ls_ts(symbol: str, period: str) -> Optional[datetime]:
    ls_dir = DATA_ROOT / symbol.upper() / "LS" / period
    if not ls_dir.exists():
        return None

    files = sorted(ls_dir.glob("*.parquet"))
    if not files:
        return None

    # 뒤에서부터 찾아서 비어있지 않은 마지막 파일 사용
    for path in reversed(files):
        try:
            df = pd.read_parquet(path, columns=["ts"])
            if not df.empty:
                return ms_to_dt(int(df["ts"].max()))
        except Exception as e:
            print(f"[warn] failed reading {path}: {e}")

    return None


def normalize_top_account_df(symbol: str, period: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=[
            "symbol", "period", "ts",
            "top_account_long_short_ratio",
            "top_account_long_pct",
            "top_account_short_pct",
        ])

    out = pd.DataFrame({
        "symbol": symbol.upper(),
        "period": period,
        "ts": raw_df["timestamp"].astype("int64"),
        "top_account_long_short_ratio": pd.to_numeric(raw_df["longShortRatio"], errors="coerce"),
        "top_account_long_pct": pd.to_numeric(raw_df["longAccount"], errors="coerce"),
        "top_account_short_pct": pd.to_numeric(raw_df["shortAccount"], errors="coerce"),
    })
    return out.sort_values("ts").drop_duplicates(subset=["symbol", "period", "ts"]).reset_index(drop=True)


def normalize_top_position_df(symbol: str, period: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=[
            "symbol", "period", "ts",
            "top_position_long_short_ratio",
            "top_position_long_pct",
            "top_position_short_pct",
        ])

    out = pd.DataFrame({
        "symbol": symbol.upper(),
        "period": period,
        "ts": raw_df["timestamp"].astype("int64"),
        "top_position_long_short_ratio": pd.to_numeric(raw_df["longShortRatio"], errors="coerce"),
        "top_position_long_pct": pd.to_numeric(raw_df["longAccount"], errors="coerce"),
        "top_position_short_pct": pd.to_numeric(raw_df["shortAccount"], errors="coerce"),
    })
    return out.sort_values("ts").drop_duplicates(subset=["symbol", "period", "ts"]).reset_index(drop=True)


def normalize_global_account_df(symbol: str, period: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=[
            "symbol", "period", "ts",
            "global_account_long_short_ratio",
            "global_account_long_pct",
            "global_account_short_pct",
        ])

    out = pd.DataFrame({
        "symbol": symbol.upper(),
        "period": period,
        "ts": raw_df["timestamp"].astype("int64"),
        "global_account_long_short_ratio": pd.to_numeric(raw_df["longShortRatio"], errors="coerce"),
        "global_account_long_pct": pd.to_numeric(raw_df["longAccount"], errors="coerce"),
        "global_account_short_pct": pd.to_numeric(raw_df["shortAccount"], errors="coerce"),
    })
    return out.sort_values("ts").drop_duplicates(subset=["symbol", "period", "ts"]).reset_index(drop=True)


def normalize_open_interest_df(symbol: str, period: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=[
            "symbol", "period", "ts",
            "sum_open_interest",
            "sum_open_interest_value",
            "cmc_circulating_supply",
        ])

    out = pd.DataFrame({
        "symbol": symbol.upper(),
        "period": period,
        "ts": raw_df["timestamp"].astype("int64"),
        "sum_open_interest": pd.to_numeric(raw_df["sumOpenInterest"], errors="coerce"),
        "sum_open_interest_value": pd.to_numeric(raw_df["sumOpenInterestValue"], errors="coerce"),
        "cmc_circulating_supply": pd.to_numeric(raw_df.get("CMCCirculatingSupply"), errors="coerce"),
    })
    return out.sort_values("ts").drop_duplicates(subset=["symbol", "period", "ts"]).reset_index(drop=True)


def merge_ls_parts(
    symbol: str,
    period: str,
    top_acc_raw: pd.DataFrame,
    top_pos_raw: pd.DataFrame,
    global_raw: pd.DataFrame,
    oi_raw: pd.DataFrame,
) -> pd.DataFrame:
    top_acc = normalize_top_account_df(symbol, period, top_acc_raw)
    top_pos = normalize_top_position_df(symbol, period, top_pos_raw)
    global_acc = normalize_global_account_df(symbol, period, global_raw)
    oi = normalize_open_interest_df(symbol, period, oi_raw)

    dfs = [top_acc, top_pos, global_acc, oi]
    non_empty = [df for df in dfs if not df.empty]

    if not non_empty:
        return pd.DataFrame(columns=[
            "symbol", "period", "ts",
            "top_account_long_short_ratio",
            "top_account_long_pct",
            "top_account_short_pct",
            "top_position_long_short_ratio",
            "top_position_long_pct",
            "top_position_short_pct",
            "global_account_long_short_ratio",
            "global_account_long_pct",
            "global_account_short_pct",
            "sum_open_interest",
            "sum_open_interest_value",
            "cmc_circulating_supply",
        ])

    merged = non_empty[0].copy()
    for df in non_empty[1:]:
        merged = merged.merge(df, on=["symbol", "period", "ts"], how="outer")

    return merged.sort_values("ts").drop_duplicates(subset=["symbol", "period", "ts"]).reset_index(drop=True)


def append_and_save_ls_by_day(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    total_saved_rows = 0

    x = df.copy()
    x["dt"] = pd.to_datetime(x["ts"], unit="ms", utc=True)
    x["date"] = x["dt"].dt.strftime("%Y-%m-%d")

    for date_str, group in x.groupby("date", sort=True):
        day_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        symbol = str(group["symbol"].iloc[0])
        period = str(group["period"].iloc[0])

        path = get_ls_day_file(symbol, period, day_dt)
        ensure_parent_dir(path)

        new_df = (
            group
            .drop(columns=["dt", "date"])
            .sort_values("ts")
            .reset_index(drop=True)
        )

        if path.exists():
            try:
                old_df = pd.read_parquet(path)
                merged = pd.concat([old_df, new_df], ignore_index=True)
                merged = (
                    merged
                    .drop_duplicates(subset=["symbol", "period", "ts"], keep="last")
                    .sort_values("ts")
                    .reset_index(drop=True)
                )
            except Exception as e:
                print(f"[warn] failed reading old parquet {path}: {e}")
                merged = (
                    new_df
                    .drop_duplicates(subset=["symbol", "period", "ts"], keep="last")
                    .sort_values("ts")
                    .reset_index(drop=True)
                )
        else:
            merged = (
                new_df
                .drop_duplicates(subset=["symbol", "period", "ts"], keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )

        merged.to_parquet(path, index=False)
        total_saved_rows += len(new_df)
        print(f"[saved] {path} incoming={len(new_df)} final={len(merged)}")

    return total_saved_rows


def empty_raw_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp"])


def fetch_window_and_save(
    symbol: str,
    period: str,
    window_start: datetime,
    window_end: datetime,
) -> int:
    start_ms = dt_to_ms(window_start)
    end_ms = dt_to_ms(window_end)

    print(
        f"[window] {symbol} {period} "
        f"{window_start.isoformat()} -> {window_end.isoformat()}"
    )

    try:
        top_acc_json = fetch_top_long_short_account_ratio(symbol, period, start_ms, end_ms, LS_LIMIT)
    except Exception as e:
        print(f"[warn] topLongShortAccountRatio failed: {e}")
        top_acc_json = []

    time.sleep(BASE_SLEEP)

    try:
        top_pos_json = fetch_top_long_short_position_ratio(symbol, period, start_ms, end_ms, LS_LIMIT)
    except Exception as e:
        print(f"[warn] topLongShortPositionRatio failed: {e}")
        top_pos_json = []

    time.sleep(BASE_SLEEP)

    try:
        global_json = fetch_global_long_short_account_ratio(symbol, period, start_ms, end_ms, LS_LIMIT)
    except Exception as e:
        print(f"[warn] globalLongShortAccountRatio failed: {e}")
        global_json = []

    time.sleep(BASE_SLEEP)

    try:
        oi_json = fetch_open_interest_hist(symbol, period, start_ms, end_ms, LS_LIMIT)
    except Exception as e:
        print(f"[warn] openInterestHist failed: {e}")
        oi_json = []

    top_acc_raw = pd.DataFrame(top_acc_json) if top_acc_json else empty_raw_df()
    top_pos_raw = pd.DataFrame(top_pos_json) if top_pos_json else empty_raw_df()
    global_raw = pd.DataFrame(global_json) if global_json else empty_raw_df()
    oi_raw = pd.DataFrame(oi_json) if oi_json else empty_raw_df()

    merged = merge_ls_parts(
        symbol=symbol,
        period=period,
        top_acc_raw=top_acc_raw,
        top_pos_raw=top_pos_raw,
        global_raw=global_raw,
        oi_raw=oi_raw,
    )

    if merged.empty:
        print("[window] no data")
        return 0

    saved_rows = append_and_save_ls_by_day(merged)
    print(f"[window] merged_rows={len(merged)} saved_rows={saved_rows}")
    return len(merged)


@dataclass
class CollectLSResult:
    symbol: str
    period: str
    requested_start: str
    actual_start: Optional[str]
    end_time: Optional[str]
    rows: int
    window_count: int
    status: str


def collect_ls_snapshot(
    symbol: str,
    period: str,
    start_time: str | int | datetime,
) -> CollectLSResult:
    symbol = symbol.upper()

    if period not in VALID_PERIODS:
        raise ValueError(f"invalid period: {period}. valid={list(VALID_PERIODS)}")

    period_ms = VALID_PERIODS[period]
    requested_start_dt = parse_start_dt(start_time, period)

    now_utc = datetime.now(timezone.utc)
    end_dt = floor_dt(now_utc, period_ms) - timedelta(milliseconds=period_ms)

    # Binance futures/data 계열은 최근 30일 제한 고려
    latest_allowed_start = floor_dt(end_dt - timedelta(days=30), period_ms)
    requested_start_dt = max(requested_start_dt, latest_allowed_start)

    last_saved = get_last_saved_ls_ts(symbol, period)
    if last_saved is not None:
        actual_start_dt = max(requested_start_dt, ms_to_dt(dt_to_ms(last_saved) + period_ms))
    else:
        actual_start_dt = requested_start_dt

    if actual_start_dt > end_dt:
        return CollectLSResult(
            symbol=symbol,
            period=period,
            requested_start=requested_start_dt.isoformat(),
            actual_start=actual_start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            rows=0,
            window_count=0,
            status="up_to_date",
        )

    window_delta = WINDOWS_BY_PERIOD[period]
    total_rows = 0
    window_count = 0

    current_start = actual_start_dt
    while current_start <= end_dt:
        current_end = min(
            current_start + window_delta - timedelta(milliseconds=1),
            end_dt,
        )

        rows = fetch_window_and_save(
            symbol=symbol,
            period=period,
            window_start=current_start,
            window_end=current_end,
        )
        total_rows += rows
        window_count += 1

        next_start = floor_dt(current_end + timedelta(milliseconds=1), period_ms)
        if next_start <= current_start:
            next_start = ms_to_dt(dt_to_ms(current_start) + period_ms)

        current_start = next_start
        time.sleep(BASE_SLEEP)

    status = "ok" if total_rows > 0 else "no_data"

    return CollectLSResult(
        symbol=symbol,
        period=period,
        requested_start=requested_start_dt.isoformat(),
        actual_start=actual_start_dt.isoformat(),
        end_time=end_dt.isoformat(),
        rows=total_rows,
        window_count=window_count,
        status=status,
    )


if __name__ == "__main__":
    result = collect_ls_snapshot(
        symbol="BTCUSDT",
        period="5m",
        start_time="2026-03-01T00:00:00+00:00",
    )
    print(result)
