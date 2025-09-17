import pandas as pd, yaml
from pathlib import Path

def _read_csv_any(path: Path):
    for enc in ('utf-8','cp949','euc-kr'):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def _normalize_news_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df=df.rename(columns={c:str(c).strip() for c in df.columns}); out=pd.DataFrame()
    if set(['date_time','ticker','title']).issubset(df.columns):
        out['date_time']=pd.to_datetime(df['date_time'], errors='coerce'); out['ticker']=df.get('ticker',ticker).fillna(ticker); out['title']=df['title'].astype(str); out['body']=df.get('body',' ').astype(str); return out.dropna(subset=['date_time','ticker','title'])
    cand_title=[c for c in df.columns if str(c).lower() in ['title','제목','0']]; cand_body=[c for c in df.columns if str(c).lower() in ['body','본문','0.1','content']]
    date_col=None
    for c in df.columns:
        s=df[c].astype(str)
        if s.str.match(r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$', na=False).sum()>len(df)*0.5:
            date_col=c; break
    if cand_title and date_col is not None:
        out['title']=df[cand_title[0]].astype(str); out['body']=df[cand_body[0]].astype(str) if cand_body else ''
        d=pd.to_datetime(df[date_col], errors='coerce'); out['date_time']=pd.to_datetime(d.dt.strftime('%Y-%m-%d')+' 09:00'); out['ticker']=ticker; return out.dropna(subset=['date_time','title'])
    if 'date' in df.columns and 'score' in df.columns:
        d=pd.to_datetime(df['date'], errors='coerce'); out=pd.DataFrame({'date_time':pd.to_datetime(d.dt.strftime('%Y-%m-%d')+' 09:00'),'ticker':ticker,'title':'Daily score','body':df['score'].astype(str)}); return out.dropna(subset=['date_time'])
    if len(df.columns)>=2:
        out['title']=df.iloc[:,0].astype(str); out['body']=df.iloc[:,1].astype(str); out['date_time']=pd.to_datetime('today').normalize(); out['ticker']=ticker; return out
    return pd.DataFrame(columns=['date_time','ticker','title','body'])

def load_news_from_repo(news_repo_root: str, mapping_yaml: str) -> pd.DataFrame:
    base=Path(news_repo_root); data_dir=base/'data'; cfg=yaml.safe_load(open(mapping_yaml,'r',encoding='utf-8'))
    folder_to_ticker=cfg.get('folder_to_ticker',{}); rows=[]
    if not data_dir.exists(): return pd.DataFrame(columns=['date_time','ticker','title','body'])
    for sub in data_dir.iterdir():
        if not sub.is_dir(): continue
        ticker=folder_to_ticker.get(sub.name); 
        if ticker is None: continue
        for csv in sub.rglob('*.csv'):
            try:
                df=_read_csv_any(csv); rows.append(_normalize_news_df(df, ticker))
            except Exception: continue
    if not rows: return pd.DataFrame(columns=['date_time','ticker','title','body'])
    out=pd.concat(rows, axis=0, ignore_index=True).dropna(subset=['date_time','ticker','title']).sort_values('date_time').reset_index(drop=True)
    return out
