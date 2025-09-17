import os, pandas as pd
from collectors.kiwoom_adapter import collect_prices
from integrations.ai_news_integration import build_panel, run_and_postprocess

STRATEGY_NAME = 'ML_NEWS_RF'

def get_buy_list_by_ml(date=None, aitb_root='.'):
    src=os.getenv('PRICE_SOURCE','auto')  # 'collector' | 'mysql' | 'auto'
    prefer='collector' if src in ('collector','auto') else 'mysql'
    prices = collect_prices(prefer=prefer)
    panel = build_panel(prices)
    _, _, out = run_and_postprocess(panel, algo='rf', thresh=float(os.getenv('ML_THRESH','0.58')), top_k=int(os.getenv('ML_TOPK','50')))
    return [(r['ticker'], float(r['weight'])) for _, r in out.iterrows()]
