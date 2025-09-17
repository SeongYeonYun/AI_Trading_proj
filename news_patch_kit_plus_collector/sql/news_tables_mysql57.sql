CREATE TABLE IF NOT EXISTS news_daily_features (
  ticker VARCHAR(16) NOT NULL,
  date DATE NOT NULL,
  news_count INT DEFAULT 0,
  sent_mean DOUBLE DEFAULT 0,
  sent_pos_ratio DOUBLE DEFAULT 0,
  sent_std DOUBLE DEFAULT 0,
  sig_count INT DEFAULT 0,
  pos_hits INT DEFAULT 0,
  neg_hits INT DEFAULT 0,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (ticker, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS signals (
  date DATE NOT NULL,
  ticker VARCHAR(16) NOT NULL,
  strategy VARCHAR(32) NOT NULL,
  model_id BIGINT NULL,
  pred_prob DOUBLE,
  weight DOUBLE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (date, ticker, strategy),
  KEY idx_date_strategy (date, strategy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
