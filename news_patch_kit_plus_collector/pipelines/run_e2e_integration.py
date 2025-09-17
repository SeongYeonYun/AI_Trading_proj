import argparse, os
from collectors.kiwoom_adapter import collect_prices
from integrations.ai_news_integration import build_panel, run_and_postprocess

def main():
    ap=argparse.ArgumentParser();
    ap.add_argument('--source', default='mysql', choices=['mysql','collector','auto'])
    ap.add_argument('--algo', default='rf'); ap.add_argument('--thresh', type=float, default=0.58); ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--out_buylist', default='data/buy_list_ml.csv'); args=ap.parse_args()
    prefer = 'collector' if args.source in ('collector','auto') else 'mysql'
    prices = collect_prices(prefer=prefer)
    panel = build_panel(prices)
    _, _, out = run_and_postprocess(panel, algo=args.algo, thresh=args.thresh, top_k=args.top_k)
    out.to_csv(args.out_buylist, index=False)
    print('Saved', args.out_buylist)
if __name__=='__main__': main()
