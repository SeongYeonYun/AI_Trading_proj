import pandas as pd
from pathlib import Path
from typing import Optional

def load_prices_from_repo(repo_root: str, start: Optional[str]=None, end: Optional[str]=None) -> pd.DataFrame:
    root=Path(repo_root); candidates=[root/'data'/'prices.csv', root/'data'/'daily_prices.csv', root/'prices.csv']
    for p in candidates:
        if p.exists(): df=pd.read_csv(p, parse_dates=['date']); break
    else: raise FileNotFoundError('Price CSV not found. Export OHLCV to data/prices.csv')
    if start: df=df[df['date']>=pd.to_datetime(start)]
    if end: df=df[df['date']<=pd.to_datetime(end)]
    need=['date','ticker','open','high','low','close','volume']; miss=[c for c in need if c not in df.columns]
    if miss: raise ValueError(f'Missing columns in price data: {miss}')
    return df.sort_values(['ticker','date']).reset_index(drop=True)
