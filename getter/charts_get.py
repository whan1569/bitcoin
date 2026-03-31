from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import requests


BINANCE_FAPI_BASE = "https://fapi.binance.com"
#DATA_ROOT = Path("data")
DATA_ROOT = Path("E:/data")
REQUEST_TIMEOUT = 15
AGG_LIMIT = 1000
MAX_RETRIES = 6
BASE_SLEEP = 0.25


def floor_to_minute(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(second=0, microsecond=0)


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def parse_start_minute(value: str | int | datetime) -> datetime:
    if isinstance(value, datetime):
        return floor_to_minute(value)

    if isinstance(value, int):
        if value > 10_000_000_000:
            dt = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(value, tz=timezone.utc)
        return floor_to_minute(dt)

    s = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return floor_to_minute(dt)


def get_day_file(symbol: str, day: datetime) -> Path:
    return DATA_ROOT / symbol.upper() / "1m" / f"{day:%Y-%m-%d}.parquet"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_last_saved_minute(symbol: str) -> Optional[datetime]:
    symbol_dir = DATA_ROOT / symbol.upper() / "1m"
    if not symbol_dir.exists():
        return None

    files = sorted(symbol_dir.glob("*.parquet"))
    if not files:
        return None

    last_file = files[-1]
    df = pd.read_parquet(last_file, columns=["ts"])
    if df.empty:
        return None

    return ms_to_dt(int(df["ts"].max()))


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


def fetch_agg_trades(symbol: str, start_ms: int, end_ms: int, limit: int = AGG_LIMIT) -> list[dict[str, Any]]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/aggTrades"
    params = {
        "symbol": symbol.upper(),
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    return safe_get_json(url, params)


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate"
    params = {
        "symbol": symbol.upper(),
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    return safe_get_json(url, params)


def fetch_all_agg_trades(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    chunk_minutes: int = 30,
) -> pd.DataFrame:
    """
    과거 aggTrades를 chunk 단위로 나눠서 안전하게 수집.
    너무 긴 구간을 한 번에 때리지 않음.
    """
    rows: list[dict[str, Any]] = []
    current_chunk_start = start_dt

    # end_dt는 "완료된 1분의 시작 시각"
    overall_end_ms = dt_to_ms(end_dt + timedelta(minutes=1)) - 1

    while current_chunk_start <= end_dt:
        current_chunk_end = min(
            current_chunk_start + timedelta(minutes=chunk_minutes) - timedelta(milliseconds=1),
            ms_to_dt(overall_end_ms),
        )

        cursor_ms = dt_to_ms(current_chunk_start)
        chunk_end_ms = dt_to_ms(current_chunk_end)

        print(
            f"[chunk] {symbol} "
            f"{current_chunk_start.isoformat()} -> {current_chunk_end.isoformat()}"
        )

        while cursor_ms <= chunk_end_ms:
            batch = fetch_agg_trades(
                symbol=symbol,
                start_ms=cursor_ms,
                end_ms=chunk_end_ms,
                limit=AGG_LIMIT,
            )

            if not batch:
                break

            for t in batch:
                rows.append(
                    {
                        "agg_id": int(t["a"]),
                        "price": float(t["p"]),
                        "qty": float(t["q"]),
                        "ts": int(t["T"]),
                    }
                )

            last_ts = int(batch[-1]["T"])
            last_agg_id = int(batch[-1]["a"])

            # 같은 타임스탬프가 연속일 가능성 방지용
            cursor_ms = last_ts + 1

            # 덜 찼으면 이 chunk는 끝
            if len(batch) < AGG_LIMIT:
                break

            # 너무 빠르게 연속 호출하지 않기
            time.sleep(BASE_SLEEP)

        current_chunk_start = floor_to_minute(current_chunk_end + timedelta(milliseconds=1))

        # chunk 사이도 살짝 쉬기
        time.sleep(BASE_SLEEP)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["agg_id"]).sort_values(["ts", "agg_id"]).reset_index(drop=True)
    return df


def build_minute_bars_from_trades(symbol: str, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    agg trades -> 1분 바 직접 생성
    """
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "ts",
                "open",
                "high",
                "low",
                "close",
                "mean",
                "path_length",
                "volume",
                "funding_rate",
                "funding_mean",
                "funding_path_length",
            ]
        )

    df = trades_df.copy()
    df["minute_ts"] = (df["ts"] // 60000) * 60000

    bars: list[dict[str, Any]] = []

    for minute_ts, g in df.groupby("minute_ts", sort=True):
        g = g.sort_values(["ts", "agg_id"]).reset_index(drop=True)

        prices = g["price"]
        qtys = g["qty"]

        path_length = float(prices.diff().abs().fillna(0.0).sum())
        volume = float(qtys.sum())
        mean = float((prices * qtys).sum() / volume) if volume > 0 else float(prices.mean())

        bars.append(
            {
                "symbol": symbol.upper(),
                "ts": int(minute_ts),
                "open": float(prices.iloc[0]),
                "high": float(prices.max()),
                "low": float(prices.min()),
                "close": float(prices.iloc[-1]),
                "mean": mean,
                "path_length": path_length,
                "volume": volume,
                "funding_rate": None,
                "funding_mean": None,
                "funding_path_length": None,
            }
        )

    return pd.DataFrame(bars).sort_values("ts").reset_index(drop=True)


def attach_funding_to_bars(symbol: str, bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    과거 funding rate를 각 1분 바에 붙인다.
    funding_mean, funding_path_length는 과거 1분 스냅샷이 없으므로
    일단 funding_rate 복사 / 0.0 처리.
    """
    if bars_df.empty:
        return bars_df

    start_ms = int(bars_df["ts"].min()) - 8 * 60 * 60 * 1000
    end_ms = int(bars_df["ts"].max())

    funding = fetch_funding_rates(symbol, start_ms, end_ms, limit=1000)

    out = bars_df.sort_values("ts").copy()

    if not funding:
        out["funding_rate"] = None
        out["funding_mean"] = None
        out["funding_path_length"] = 0.0
        return out

    fdf = pd.DataFrame(
        {
            "funding_ts": [int(x["fundingTime"]) for x in funding],
            "funding_rate_raw": [float(x["fundingRate"]) for x in funding],
        }
    ).sort_values("funding_ts")

    merged = pd.merge_asof(
        out[["ts"]],
        fdf,
        left_on="ts",
        right_on="funding_ts",
        direction="backward",
    )

    out["funding_rate"] = merged["funding_rate_raw"].values
    out["funding_mean"] = out["funding_rate"]
    out["funding_path_length"] = 0.0
    return out


def append_and_save_by_day(df: pd.DataFrame) -> None:
    if df.empty:
        return

    x = df.copy()
    x["dt"] = pd.to_datetime(x["ts"], unit="ms", utc=True)
    x["date"] = x["dt"].dt.strftime("%Y-%m-%d")

    for date_str, group in x.groupby("date", sort=True):
        day_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        path = get_day_file(group["symbol"].iloc[0], day_dt)
        ensure_parent_dir(path)

        new_df = group.drop(columns=["dt", "date"]).sort_values("ts").reset_index(drop=True)

        if path.exists():
            old_df = pd.read_parquet(path)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged = (
                merged.drop_duplicates(subset=["symbol", "ts"], keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )
        else:
            merged = new_df

        merged.to_parquet(path, index=False)
        print(f"[saved] {path} rows={len(merged)}")


@dataclass
class CollectResult:
    symbol: str
    requested_start: str
    actual_start: Optional[str]
    end_minute: Optional[str]
    trade_rows: int
    bar_rows: int
    status: str


def collect_minute_bars(
    symbol: str,
    start_minute: str | int | datetime,
    chunk_minutes: int = 30,
) -> CollectResult:
    symbol = symbol.upper()
    requested_start_dt = parse_start_minute(start_minute)

    last_saved = get_last_saved_minute(symbol)
    if last_saved is not None:
        actual_start = max(requested_start_dt, last_saved + timedelta(minutes=1))
    else:
        actual_start = requested_start_dt

    now_utc = datetime.now(timezone.utc)
    end_minute = floor_to_minute(now_utc) - timedelta(minutes=1)

    if actual_start > end_minute:
        return CollectResult(
            symbol=symbol,
            requested_start=requested_start_dt.isoformat(),
            actual_start=actual_start.isoformat(),
            end_minute=end_minute.isoformat(),
            trade_rows=0,
            bar_rows=0,
            status="up_to_date",
        )

    trades_df = fetch_all_agg_trades(
        symbol=symbol,
        start_dt=actual_start,
        end_dt=end_minute,
        chunk_minutes=chunk_minutes,
    )

    if trades_df.empty:
        return CollectResult(
            symbol=symbol,
            requested_start=requested_start_dt.isoformat(),
            actual_start=actual_start.isoformat(),
            end_minute=end_minute.isoformat(),
            trade_rows=0,
            bar_rows=0,
            status="no_trades",
        )

    bars_df = build_minute_bars_from_trades(symbol, trades_df)
    bars_df = attach_funding_to_bars(symbol, bars_df)
    append_and_save_by_day(bars_df)

    return CollectResult(
        symbol=symbol,
        requested_start=requested_start_dt.isoformat(),
        actual_start=actual_start.isoformat(),
        end_minute=end_minute.isoformat(),
        trade_rows=len(trades_df),
        bar_rows=len(bars_df),
        status="ok",
    )


if __name__ == "__main__":
    # 예시:
    # 이미 저장된 데이터가 있으면 마지막 저장분 다음부터 이어서 수집
    result = collect_minute_bars(
        symbol="BTCUSDT",
        start_minute="2026-03-15T00:00:00+00:00",
        chunk_minutes=30,   # 15~30 추천
    )
    print(result)
