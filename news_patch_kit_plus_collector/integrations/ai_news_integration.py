import os, pandas as pd
from configs.cost import Cost
from models.classical import ModelCfg, train_eval
from ai_news.news_features import build_news_daily_features
from integrations.news_loader import load_news_from_repo
from library.signal_adapter import apply_postprocess

NEWS_ROOT=os.getenv('NEWS_REPO_ROOT','./Text_Mining_NewsData')
NEWS_MAP = os.getenv('NEWS_MAP_YML','configs/news_mapping.yml')
LEXICON  = os.getenv('NEWS_LEXICON','ai_news/lexicon.yml')

def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    news_raw = load_news_from_repo(NEWS_ROOT, NEWS_MAP)
    news_daily = build_news_daily_features(news_raw, LEXICON)
    news_daily['date'] = pd.to_datetime(news_daily['date'])
    panel = prices.merge(news_daily, on=['ticker','date'], how='left').fillna({
        'news_count':0,'sent_mean':0,'sent_pos_ratio':0,'sent_std':0,'sig_count':0,'pos_hits':0,'neg_hits':0
    })
    return panel

def run_and_postprocess(panel: pd.DataFrame, algo='rf', thresh=0.58, top_k=50):
    cfg = ModelCfg(algo=algo, thresh=thresh, use_news=True)
    metrics, preds = train_eval(panel, cfg, Cost())
    latest_date = preds['date'].max()
    latest = preds[preds['date']==latest_date][['date','ticker','pred_prob','close','atr14']].copy()
    latest['value_traded_ma20'] = latest['close']*1e6
    out = apply_postprocess(latest, df_index=None, df_sector=None, today=latest_date, top_k=top_k)
    return metrics, preds, out
