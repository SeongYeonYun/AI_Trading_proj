import pandas as pd, numpy as np

def apply_postprocess(df_signals, df_index=None, df_sector=None, today=None, min_value_traded=3e8, max_atr_pct=0.08, top_k=50, per_sector_cap=3, target_port_vol=0.01):
    df=df_signals.copy().sort_values(['pred_prob','ticker'], ascending=[False,True])
    if {'value_traded_ma20'}.issubset(df.columns): df=df[df['value_traded_ma20']>=min_value_traded]
    if {'atr14','close'}.issubset(df.columns):
        df['atr_pct']=(df['atr14']/df['close']).clip(lower=0); df=df[df['atr_pct']<=max_atr_pct]
    if df_sector is not None and not df_sector.empty:
        df=df.merge(df_sector,on='ticker',how='left'); out=[]; cap={}
        for _,r in df.iterrows():
            s=r.get('sector','UNKNOWN');
            if cap.get(s,0)<per_sector_cap: out.append(r); cap[s]=cap.get(s,0)+1
        df=pd.DataFrame(out)
    df=df.head(top_k).copy()
    if 'atr14' in df.columns and 'close' in df.columns and not df.empty:
        df['atr_pct']=(df['atr14']/df['close']).replace(0, np.nan)
        inv=1.0/df['atr_pct']; inv=inv.fillna(inv.median()); w=inv/inv.sum(); w=w.clip(upper=0.10); w=w/w.sum(); df['weight']=w.values
    else:
        df['weight']=1.0/max(len(df),1)
    keep=[c for c in ['date','ticker','pred_prob','weight','atr_pct','value_traded_ma20','sector'] if c in df.columns]
    return df[keep].reset_index(drop=True)
