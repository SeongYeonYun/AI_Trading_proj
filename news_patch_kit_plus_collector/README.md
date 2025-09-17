# News Patch Kit + Collector (Full)
- Generated: 2025-09-17 02:49:42
- Includes ALL source files, patch diff, SQL, configs, integrations, plugins, pipelines.

## Quickstart
```
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python pipelines/run_e2e_integration.py --source auto --algo rf --thresh 0.58 --top_k 50
```

## Apply Patch to existing repo
```
git apply news_integration.patch
```
If patch fails (context mismatch), insert the strategy block manually per the diff.

