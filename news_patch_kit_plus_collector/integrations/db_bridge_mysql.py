import pandas as pd, pymysql, yaml
from configs.db_mysql import MYSQL
from pathlib import Path

def load_prices_from_mysql(schema_yml='configs/schema.yml', start=None, end=None):
    cfg=yaml.safe_load(open(schema_yml,'r',encoding='utf-8'))
    table=cfg['ohlcv_daily_table']; c=cfg['columns']
    sql=f"""
    SELECT {c['date']} AS date,
           {c['ticker']} AS ticker,
           {c['open']}  AS open,
           {c['high']}  AS high,
           {c['low']}   AS low,
           {c['close']} AS close,
           {c['volume']} AS volume
    FROM {table}
    WHERE (%s IS NULL OR {c['date']} >= %s)
      AND (%s IS NULL OR {c['date']} <= %s)
    ORDER BY {c['ticker']}, {c['date']}
    """
    conn=pymysql.connect(**MYSQL)
    try:
        df=pd.read_sql(sql, conn, params=[start,start,end,end], parse_dates=['date'])
    finally:
        conn.close()
    return df
