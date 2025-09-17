import pandas as pd
import numpy as np

def market_regime_filter(df_index, today, ma_window=200, reduce_ratio=0.5):
    di = df_index.sort_values("date").copy()
    di["ma"] = di["close"].rolling(ma_window, min_periods=ma_window).mean()
    row = di[di["date"] <= today].tail(1)
    if row.empty:
        return 1.0
    below = row["close"].values[0] < row["ma"].values[0]
    return reduce_ratio if below else 1.0

def apply_postprocess(
    df_signals,
    df_index=None,
    df_sector=None,
    today=None,
    min_value_traded=3e8,
    max_atr_pct=0.08,
    top_k=50,
    per_sector_cap=3,
    target_port_vol=0.01
):
    df = df_signals.copy().sort_values(["pred_prob","ticker"], ascending=[False, True])

    if {"value_traded_ma20"}.issubset(df.columns):
        df = df[df["value_traded_ma20"] >= min_value_traded].copy()

    if {"atr14","close"}.issubset(df.columns):
        df["atr_pct"] = (df["atr14"] / df["close"]).clip(lower=0)
        df = df[df["atr_pct"] <= max_atr_pct].copy()

    if df_sector is not None and not df_sector.empty:
        df = df.merge(df_sector, on="ticker", how="left")
        out_rows, per_sec_count = [], {}
        for _, r in df.iterrows():
            sec = r.get("sector", "UNKNOWN")
            if per_sec_count.get(sec, 0) < per_sector_cap:
                out_rows.append(r); per_sec_count[sec] = per_sec_count.get(sec, 0) + 1
        df = pd.DataFrame(out_rows)

    df = df.head(top_k).copy()

    exposure_ratio = 1.0
    if df_index is not None and today is not None:
        exposure_ratio = market_regime_filter(df_index, today, ma_window=200, reduce_ratio=0.5)

    if "atr14" in df.columns and "close" in df.columns and not df.empty:
        df["atr_pct"] = (df["atr14"] / df["close"]).replace(0, np.nan)
        inv_vol = 1.0 / df["atr_pct"]
        inv_vol = inv_vol.fillna(inv_vol.median())
        w = inv_vol / inv_vol.sum()
        w = w * exposure_ratio
        w = w.clip(upper=0.10)
        w = w / w.sum() * exposure_ratio
        df["weight"] = w.values
    else:
        df["weight"] = exposure_ratio / max(len(df), 1)

    keep = [c for c in ["date","ticker","pred_prob","weight","atr_pct","value_traded_ma20","sector"] if c in df.columns]
    return df[keep].reset_index(drop=True)
