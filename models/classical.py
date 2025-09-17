import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from features.ta import add_ta
from configs.cost import Cost

BASE_FEATS = ["bb_pct","macd","macd_sig","macd_hist","rsi14","ret1","atr14","vol_chg"]
NEWS_FEATS = ["news_count","sent_mean","sent_pos_ratio","sent_std","sig_count","pos_hits","neg_hits"]

@dataclass
class ModelCfg:
    algo: str = "rf" # "rf" or "svm" or "lr_news"
    rf_n: int = 500
    rf_max_depth: int | None = None
    svm_c: float = 2.0
    svm_gamma: str | float = "scale"
    svm_kernel: str = "rbf"
    thresh: float = 0.55
    use_news: bool = True

def _label_with_cost(g: pd.DataFrame, cost: Cost) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    next_close = g["close"].shift(-1)
    gross = (next_close / g["close"]) - 1.0
    g["ret_net_t1"] = gross - (cost.buy_fee + cost.sell_fee + cost.sell_tax)
    g["y"] = (g["ret_net_t1"] > 0).astype(int)
    return g

def build_set(df: pd.DataFrame, cost: Cost, use_news: bool = True) -> pd.DataFrame:
    df = df.sort_values(["ticker","date"])
    df = df.groupby("ticker", group_keys=False).apply(add_ta)
    df = df.groupby("ticker", group_keys=False).apply(_label_with_cost, cost=cost)
    # fill news feats if missing
    for c in NEWS_FEATS:
        if c not in df.columns:
            df[c] = 0.0
    feats = BASE_FEATS + (NEWS_FEATS if use_news else [])
    return df.dropna(subset=feats + ["y"])

def walk_splits(dates: pd.Series, n=5, min_train=252*2, val=63, gap=5):
    d = pd.Series(pd.to_datetime(dates.unique())).sort_values()
    out = []
    i = min_train
    while i + gap + val < len(d) and len(out) < n:
        out.append((d.iloc[:i], d.iloc[i+gap:i+gap+val]))
        i += val
    return out

def _make_clf(cfg: ModelCfg):
    if cfg.algo == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.rf_n, max_depth=cfg.rf_max_depth,
            n_jobs=-1, class_weight="balanced_subsample", random_state=42
        ), None
    elif cfg.algo == "svm":
        return SVC(C=cfg.svm_c, gamma=cfg.svm_gamma, kernel=cfg.svm_kernel,
                   probability=True, class_weight="balanced", random_state=42), StandardScaler()
    else:
        # logistic regression (news-friendly)
        return LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.3, max_iter=500,
            class_weight="balanced", n_jobs=-1
        ), StandardScaler()

def train_eval(df: pd.DataFrame, cfg: ModelCfg, cost: Cost):
    df = build_set(df, cost, use_news=cfg.use_news)
    feats = BASE_FEATS + (NEWS_FEATS if cfg.use_news else [])
    metrics, preds = [], []
    splits = walk_splits(df["date"])
    clf, scaler = _make_clf(cfg)
    for k,(trd,vld) in enumerate(splits,1):
        tr = df[df["date"].isin(trd)]
        vl = df[df["date"].isin(vld)]

        X_tr, y_tr = tr[feats].values, tr["y"].values
        X_vl, y_vl = vl[feats].values, vl["y"].values

        if scaler is not None:
            X_tr = scaler.fit_transform(X_tr)
            X_vl = scaler.transform(X_vl)

        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_vl)[:,1]
        pred = (proba >= cfg.thresh).astype(int)

        vl_bt = vl.copy()
        vl_bt["signal"] = pred
        vl_bt["strategy_ret"] = vl_bt["ret_net_t1"] * vl_bt["signal"]
        cumret = (1 + vl_bt.groupby("date")["strategy_ret"].mean()).prod() - 1

        metrics.append({"fold":k,
                        "auc": roc_auc_score(y_vl,proba),
                        "f1":  f1_score(y_vl,pred),
                        "acc": accuracy_score(y_vl,pred),
                        "cumret": cumret})
        preds.append(vl.assign(pred_prob=proba, pred=pred))
    import pandas as pd
    return pd.DataFrame(metrics), pd.concat(preds, axis=0)

def infer_latest(df_latest: pd.DataFrame, cfg: ModelCfg, cost: Cost):
    data = build_set(df_latest, cost, use_news=cfg.use_news)
    feats = BASE_FEATS + (NEWS_FEATS if cfg.use_news else [])
    X = data[feats].values
    clf, scaler = _make_clf(cfg)
    if scaler is not None:
        X = scaler.fit_transform(X)
    clf.fit(data[feats], data["y"])
    proba = clf.predict_proba(data[feats])[:,1]
    return data.assign(pred_prob=proba, pred=(proba>=cfg.thresh).astype(int))
