import re
import pandas as pd
from collections import Counter
import yaml
from datetime import time

def load_lexicon(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", str(text))
    toks = [t for t in text.split() if len(t) > 1]
    return toks

def _apply_modifiers(score, toks, idx, lex):
    # very light heuristic: if nearby intensify/soften then scale
    window = 3
    start, end = max(0, idx-window), min(len(toks), idx+window+1)
    neigh = set(toks[start:end])
    if any(w in neigh for w in lex.get("modifiers", {}).get("intensify", [])):
        score *= 1.2
    if any(w in neigh for w in lex.get("modifiers", {}).get("soften", [])):
        score *= 0.8
    if any(w in neigh for w in lex.get("negations", [])):
        score *= -1.0
    return score

def score_article(title: str, body: str|None, lex: dict) -> dict:
    txt = f"{title or ''} {body or ''}"
    toks = _tokenize(txt)
    # build dicts for quick lookup
    pos = {d["term"]: float(d["weight"]) for d in lex.get("positive", [])}
    neg = {d["term"]: float(d["weight"]) for d in lex.get("negative", [])}
    score_sum, pos_hits, neg_hits = 0.0, 0, 0
    for i, tok in enumerate(toks):
        if tok in pos:
            pos_hits += 1
            score_sum += _apply_modifiers(pos[tok], toks, i, lex)
        if tok in neg:
            neg_hits += 1
            score_sum += _apply_modifiers(neg[tok], toks, i, lex)
    total_hits = pos_hits + neg_hits
    score_norm = 0.0 if total_hits == 0 else max(-1.0, min(1.0, score_sum / total_hits / 2.0))
    return {"pos_hits": pos_hits, "neg_hits": neg_hits, "sent_score": score_norm, "has_signal": total_hits > 0}

def build_news_daily_features(df_news: pd.DataFrame, lexicon_path: str, market_close="15:30") -> pd.DataFrame:
    """Aggregate news to (ticker,date) daily features with after-hours handling.
    df_news columns expected: ['date_time','ticker','title','body']
    market_close: 'HH:MM' local time of KRX close (default 15:30)
    """
    lex = load_lexicon(lexicon_path)
    df = df_news.copy()
    dt = pd.to_datetime(df["date_time"])
    df["date_time"] = dt
    # same-day vs after-close: if time > close -> shift to next business day (approx by +1 day; align with price calendar externally)
    close_h, close_m = map(int, market_close.split(":"))
    after_close = dt.dt.time > time(close_h, close_m)
    df["date"] = dt.dt.date
    df.loc[after_close, "date"] = (dt[after_close] + pd.Timedelta(days=1)).dt.date

    feats = df.apply(lambda r: score_article(r.get("title",""), r.get("body",""), lex), axis=1, result_type="expand")
    df = pd.concat([df, feats], axis=1)

    g = df.groupby(["ticker","date"])
    out = g.agg(
        news_count=("title","count"),
        sent_mean=("sent_score","mean"),
        sent_pos_ratio=(lambda x: (x > 0).mean()),
        sent_std=("sent_score","std"),
        sig_count=("has_signal","sum"),
        pos_hits=("pos_hits","sum"),
        neg_hits=("neg_hits","sum"),
    ).reset_index()
    out["sent_std"] = out["sent_std"].fillna(0.0)
    return out
