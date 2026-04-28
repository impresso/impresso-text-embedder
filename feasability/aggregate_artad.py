"""Stream the AGGREGATED jsonl, keep only tp ∈ {article, ad}, bucket finer.

Buckets:
  ocrqa: <0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, >=0.8, NA
  len:   <200, 200-400, 400-600, 600-800, >=800

Output: tmp/aggregated_summary_artad.parquet
Columns: decade, lg, tp, len_bucket, ocrqa_bucket, n
"""
import json, time
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

JSONL = Path('115-canonical-processed-final-langident-langident-lid-ensemble_multilingual_v2-0-2__AGGREGATED.jsonl')
OUT   = Path('aggregated_summary_artad.csv')

KEEP_TP = {'article', 'ad'}

def ocrqa_bucket(q):
    if q is None: return 'NA'
    if q < 0.5:   return '<0.5'
    if q < 0.6:   return '0.5-0.6'
    if q < 0.7:   return '0.6-0.7'
    if q < 0.8:   return '0.7-0.8'
    return '>=0.8'

def len_bucket(L):
    if L < 200: return '<200'
    if L < 400: return '200-400'
    if L < 600: return '400-600'
    if L < 800: return '600-800'
    return '>=800'

def main():
    counter: Counter = Counter()
    bad = 0
    kept = 0
    seen = 0
    total = JSONL.stat().st_size
    t0 = time.time()
    with open(JSONL, 'rb', buffering=4 * 1024 * 1024) as f, \
         tqdm(total=total, unit='B', unit_scale=True, desc='scan') as pbar:
        for line in f:
            pbar.update(len(line))
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                seen += 1
                tp = d.get('tp')
                if tp not in KEEP_TP:
                    continue
                year = d.get('year')
                decade = (int(year) // 10) * 10 if year else -1
                lg = d.get('lg') or 'NA'
                length = d.get('len') or 0
                ob = ocrqa_bucket(d.get('ocrqa'))
                lb = len_bucket(length)
                counter[(decade, lg, tp, lb, ob)] += 1
                kept += 1
            except Exception:
                bad += 1
    dt = time.time() - t0
    print(f'seen: {seen:,}  kept (article/ad): {kept:,}  bad: {bad:,}  elapsed: {dt:.1f}s  ({seen/dt/1e6:.2f} M lines/s)')
    rows = [(*k, v) for k, v in counter.items()]
    df = pd.DataFrame(rows, columns=['decade', 'lg', 'tp', 'len_bucket', 'ocrqa_bucket', 'n'])
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT}  ({len(df):,} rows, {df["n"].sum():,} items)')

if __name__ == '__main__':
    main()
