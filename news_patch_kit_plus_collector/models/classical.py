import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from features.ta import add_ta
from configs.cost import Cost
BASE_FEATS=['bb_pct','macd','macd_sig','macd_hist','rsi14','ret1','atr14','vol_chg']
NEWS_FEATS=['news_count','sent_mean','sent_pos_ratio','sent_std','sig_count','pos_hits','neg_hits']
@dataclass
class ModelCfg:
    algo:str='rf'; rf_n:int=300; rf_max_depth:int|None=None; svm_c:float=2.0; svm_gamma:str|float='scale'; svm_kernel:str='rbf'; thresh:float=0.58; use_news:bool=True

def _label_with_cost(g:pd.DataFrame,c:Cost)->pd.DataFrame:
    g=g.sort_values('date').copy(); nxt=g['close'].shift(-1); gross=(nxt/g['close'])-1.0; g['ret_net_t1']=gross-(c.buy_fee+c.sell_fee+c.sell_tax); g['y']=(g['ret_net_t1']>0).astype(int); return g

def build_set(df:pd.DataFrame, c:Cost, use_news=True)->pd.DataFrame:
    df=df.sort_values(['ticker','date']); df=df.groupby('ticker',group_keys=False).apply(add_ta); df=df.groupby('ticker',group_keys=False).apply(_label_with_cost,c=c)
    for col in NEWS_FEATS:
        if col not in df.columns: df[col]=0.0
    feats=BASE_FEATS+(NEWS_FEATS if use_news else [])
    return df.dropna(subset=feats+['y'])

def walk_splits(dates:pd.Series,n=5,min_train=252*2,val=63,gap=5):
    d=pd.Series(pd.to_datetime(dates.unique())).sort_values(); out=[]; i=min_train
    while i+gap+val<len(d) and len(out)<n:
        out.append((d.iloc[:i], d.iloc[i+gap:i+gap+val])); i+=val
    return out

def _make_clf(cfg:ModelCfg):
    if cfg.algo=='rf': return RandomForestClassifier(n_estimators=cfg.rf_n,max_depth=cfg.rf_max_depth,n_jobs=-1,class_weight='balanced_subsample',random_state=42), None
    if cfg.algo=='svm': return SVC(C=cfg.svm_c,gamma=cfg.svm_gamma,kernel=cfg.svm_kernel,probability=True,class_weight='balanced',random_state=42), StandardScaler()
    return LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.3,max_iter=500,class_weight='balanced',n_jobs=-1), StandardScaler()

def train_eval(df:pd.DataFrame, cfg:ModelCfg, c:Cost):
    df=build_set(df,c,use_news=cfg.use_news); feats=BASE_FEATS+(NEWS_FEATS if cfg.use_news else [])
    metrics=[]; preds=[]; splits=walk_splits(df['date'])
    for k,(trd,vld) in enumerate(splits,1):
        tr=df[df['date'].isin(trd)]; vl=df[df['date'].isin(vld)]
        X_tr,y_tr=tr[feats].values, tr['y'].values; X_vl,y_vl=vl[feats].values, vl['y'].values
        clf,sc=_make_clf(cfg); 
        if sc is not None: X_tr=sc.fit_transform(X_tr); X_vl=sc.transform(X_vl)
        clf.fit(X_tr,y_tr); p=clf.predict_proba(X_vl)[:,1]; pred=(p>=cfg.thresh).astype(int)
        vl_bt=vl.copy(); vl_bt['signal']=pred; vl_bt['strategy_ret']=vl_bt['ret_net_t1']*vl_bt['signal']
        cum=(1+vl_bt.groupby('date')['strategy_ret'].mean()).prod()-1
        metrics.append({'fold':k,'auc':roc_auc_score(y_vl,p),'f1':f1_score(y_vl,pred),'acc':accuracy_score(y_vl,pred),'cumret':cum}); preds.append(vl.assign(pred_prob=p,pred=pred))
    return pd.DataFrame(metrics), pd.concat(preds,axis=0)
