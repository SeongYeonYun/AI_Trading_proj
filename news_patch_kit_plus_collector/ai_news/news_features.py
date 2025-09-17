import re, yaml, pandas as pd

def load_lexicon(p):
    return yaml.safe_load(open(p, 'r', encoding='utf-8'))

def _tok(t):
    t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", str(t));
    return [w for w in t.split() if len(w)>1]

def score_article(title, body, lex):
    toks = _tok(f"{title} {body}"); pos={d['term']:d['weight'] for d in lex.get('positive',[])}; neg={d['term']:d['weight'] for d in lex.get('negative',[])}
    s=0; ph=0; nh=0
    for w in toks:
        if w in pos: s+=pos[w]; ph+=1
        if w in neg: s+=neg[w]; nh+=1
    total=ph+nh
    return {'pos_hits':ph,'neg_hits':nh,'sent_score':0.0 if total==0 else max(-1.0,min(1.0,s/total/2.0)),'has_signal':total>0}

def build_news_daily_features(df, lexicon_path, market_close='15:30'):
    lex=load_lexicon(lexicon_path); df=df.copy(); dt=pd.to_datetime(df['date_time']); df['date_time']=dt
    h,m=map(int,market_close.split(':')); after=(dt.dt.hour*60+dt.dt.minute)>(h*60+m)
    df['date']=dt.dt.date; df.loc[after,'date']=(dt[after]+pd.Timedelta(days=1)).dt.date
    feats=df.apply(lambda r: score_article(r.get('title',''), r.get('body',''), lex), axis=1, result_type='expand'); df=pd.concat([df,feats],axis=1)
    g=df.groupby(['ticker','date'])
    out=g.agg(news_count=('title','count'),sent_mean=('sent_score','mean'),sent_pos_ratio=(lambda x:(x>0).mean()),sent_std=('sent_score','std'),sig_count=('has_signal','sum'),pos_hits=('pos_hits','sum'),neg_hits=('neg_hits','sum')).reset_index()
    out['sent_std']=out['sent_std'].fillna(0.0)
    return out
