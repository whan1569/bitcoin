from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


DATA_ROOT = Path("data")
SYMBOL = "XRPUSDT"
TIMEFRAME = "1m"


def load_symbol_data(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    folder = DATA_ROOT / symbol / TIMEFRAME
    files = sorted(folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)

    if start:
        start_dt = pd.Timestamp(start, tz="UTC")
        df = df[df["dt"] >= start_dt]

    if end:
        end_dt = pd.Timestamp(end, tz="UTC")
        df = df[df["dt"] <= end_dt]

    return df.reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # funding_rate 정리
    if "funding_rate" in x.columns:
        x["funding_rate"] = pd.to_numeric(x["funding_rate"], errors="coerce")

    # 이동 평균들
    if "path_length" in x.columns:
        x["path_length_ma_30"] = x["path_length"].rolling(30, min_periods=1).mean()

    if "volume" in x.columns:
        x["volume_ma_30"] = x["volume"].rolling(30, min_periods=1).mean()

    if "close" in x.columns:
        x["close_ma_30"] = x["close"].rolling(30, min_periods=1).mean()

    return x


def plot_data(df: pd.DataFrame, symbol: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # 1️⃣ 가격
    axes[0].plot(df["dt"], df["close"], label="close")
    if "close_ma_30" in df.columns:
        axes[0].plot(df["dt"], df["close_ma_30"], label="MA30")
    axes[0].set_title(f"{symbol} Price")
    axes[0].legend()
    axes[0].grid(True)

    # 2️⃣ path_length
    axes[1].plot(df["dt"], df["path_length"], label="path_length")
    if "path_length_ma_30" in df.columns:
        axes[1].plot(df["dt"], df["path_length_ma_30"], label="MA30")
    axes[1].set_title("Path Length")
    axes[1].legend()
    axes[1].grid(True)

    # 3️⃣ 거래량
    axes[2].bar(df["dt"], df["volume"], width=0.0008, label="volume")
    if "volume_ma_30" in df.columns:
        axes[2].plot(df["dt"], df["volume_ma_30"], label="MA30")
    axes[2].set_title("Volume")
    axes[2].legend()
    axes[2].grid(True)

    # 4️⃣ 펀딩비
    if "funding_rate" in df.columns:
        axes[3].plot(df["dt"], df["funding_rate"], label="funding_rate")
        axes[3].axhline(0, linestyle="--")
        axes[3].set_title("Funding Rate")
        axes[3].legend()
        axes[3].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 🔥 여기 범위 조절 (너무 길면 렉 걸림)
    df = load_symbol_data(
        SYMBOL,
        start="2026-03-19 00:00:00",  # 최근 하루 추천
        end=None,
    )

    df = add_derived_columns(df)

    print(df.tail())  # 확인용

    plot_data(df, SYMBOL)
