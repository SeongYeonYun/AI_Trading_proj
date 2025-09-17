# News Patch Kit + Collector Adapter

- Kiwoom collector가 이미 레포에 있을 때: `collectors/kiwoom_adapter.py`가 자동으로 import 시도합니다.
- collector를 직접 못 찾으면 MySQL(`configs/db_mysql.py`, `configs/schema.yml`)에서 OHLCV를 읽어 사용합니다.

## 실행
```
pip install pandas numpy scikit-learn PyYAML pymysql
python pipelines/run_e2e_integration.py --source auto --algo rf --thresh 0.58
```

## 시뮬/트레이더
- 전략명: `ML_NEWS_RF`
- `plugins/strategy_ml_news.py`를 불러 분기 추가하세요.
