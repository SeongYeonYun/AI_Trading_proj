import re
import pandas as pd
import numpy as np
from collections import Counter

def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", str(text))
    toks = [t for t in text.split() if len(t) > 1]
    return list(set(toks))  # document-level presence

def pmi_expand(df_news: pd.DataFrame, seed_words: list[str], min_df: int = 5, top_k: int = 200) -> pd.DataFrame:
    """Return candidate terms ranked by PMI with the seed set."""
    df = df_news.copy()
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna(""))
    docs = df["text"].tolist()
    tokens = [ _tokenize(t) for t in docs ]
    N = len(tokens)
    seed = set(seed_words)

    # freq of terms
    wc = Counter()
    seed_docs = 0
    for toks in tokens:
        wc.update(toks)
        if seed & set(toks):
            seed_docs += 1

    # co-occurrence with seed
    co = Counter()
    for toks in tokens:
        s = set(toks)
        if seed & s:
            co.update(s)

    rows = []
    for term, cxy in co.items():
        if term in seed:
            continue
        dfreq = wc[term]
        if dfreq < min_df or dfreq > 0.5 * N:
            continue
        px = dfreq / N
        py = seed_docs / N if seed_docs else 1e-9
        pxy = cxy / N
        if px > 0 and py > 0 and pxy > 0:
            pmi = float(np.log(pxy / (px * py)))
            rows.append((term, pmi, dfreq, cxy))
    out = pd.DataFrame(rows, columns=["term","pmi","doc_freq","co_freq"]).sort_values("pmi", ascending=False)
    return out.head(top_k)
