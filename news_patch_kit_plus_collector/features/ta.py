import pandas as pd, numpy as np

def add_ta(g: pd.DataFrame)->pd.DataFrame:
    g=g.sort_values('date').copy(); c,h,l,v=g['close'],g['high'],g['low'],g['volume']
    ma20=c.rolling(20,min_periods=20).mean(); std20=c.rolling(20,min_periods=20).std()
    g['bb_pct']=(c-(ma20-2*std20))/((ma20+2*std20)-(ma20-2*std20))
    ema12,ema26=c.ewm(span=12,adjust=False).mean(),c.ewm(span=26,adjust=False).mean()
    macd=ema12-ema26; macd_sig=macd.ewm(span=9,adjust=False).mean(); g['macd'],g['macd_sig'],g['macd_hist']=macd,macd_sig,macd-macd_sig
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); down=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rs=up/(down+1e-12)
    g['rsi14']=100-(100/(1+rs)); g['ret1']=c.pct_change(); g['atr14']=(h-l).rolling(14,min_periods=14).mean(); g['vol_chg']=v.pct_change(); return g
