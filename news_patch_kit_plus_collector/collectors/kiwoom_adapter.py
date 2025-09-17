import importlib, os, pandas as pd
from integrations.db_bridge_mysql import load_prices_from_mysql

POSSIBLE_MODULES=['collector_v3','collectorV3','collector','kiwoom_collector']
POSSIBLE_FUNCS=['get_ohlcv_panel','fetch_ohlcv_panel','download_ohlcv','get_daily_ohlcv']

def get_prices_from_existing_collector(start=None, end=None, codes=None):
    for m in POSSIBLE_MODULES:
        try:
            mod=importlib.import_module(m)
        except Exception:
            continue
        for fn in POSSIBLE_FUNCS:
            f=getattr(mod, fn, None)
            if callable(f):
                try:
                    df=f(start=start, end=end, codes=codes)
                    if isinstance(df, pd.DataFrame):
                        return df
                except TypeError:
                    try:
                        df=f(start, end, codes)
                        if isinstance(df, pd.DataFrame): return df
                    except Exception:
                        continue
                except Exception:
                    continue
    return None

def collect_prices(start=None, end=None, codes=None, prefer='collector'): 
    if prefer=='collector':
        df=get_prices_from_existing_collector(start, end, codes)
        if df is not None:
            return df
        return load_prices_from_mysql(start=start, end=end)
    else:
        try:
            return load_prices_from_mysql(start=start, end=end)
        except Exception:
            df=get_prices_from_existing_collector(start, end, codes)
            if df is not None:
                return df
            raise RuntimeError('Neither MySQL nor collector_v3 produced prices. Check paths and DB.')
