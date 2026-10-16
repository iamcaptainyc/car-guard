import json
import sqlite3
import sys
from pathlib import Path

db_path = r"C:\Users\chanceyu\Downloads\guard_contention_profile.sqlite"
json_path = r"C:\Users\chanceyu\Downloads\contention_disentangle_results.json"

if Path(json_path).exists():
    print("=== JSON RESULTS ===")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(json.dumps(data, indent=2, ensure_ascii=False))

if Path(db_path).exists():
    print("\n=== SQLITE DATABASE ANALYSIS ===")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    print("Tables found:", len(tables))
    for t in tables:
        if any(k in t.lower() for k in ["kernel", "cuda", "nvtx", "stream", "string", "event", "composite"]):
            count = cur.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            print(f"  - {t}: {count} rows")

    # 查看具体的内核事件表
    for cand in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_KERNEL_EVENTS", "TARGET_INFO_CUDA_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"]:
        if cand in tables:
            print(f"\nSchema for {cand}:")
            cols = cur.execute(f"PRAGMA table_info({cand});").fetchall()
            for c in cols:
                print(f"    {c[1]} ({c[2]})")
            print(f"Sample 3 rows from {cand}:")
            rows = cur.execute(f"SELECT * FROM {cand} LIMIT 3;").fetchall()
            for r in rows:
                print("   ", r[:8])
