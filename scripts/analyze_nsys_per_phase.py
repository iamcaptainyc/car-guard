import sqlite3
import statistics

db_path = r"C:\Users\chanceyu\Downloads\guard_contention_profile.sqlite"
con = sqlite3.connect(db_path)
cur = con.cursor()

# 找出 4 个测试阶段的时间区间
# 根据 Stream 149 (VLM) 上的 Prefill / GDN 内核出现的位置
query_prefills = """
SELECT k.start, k.end, (k.end - k.start) / 1e6 as dur_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
WHERE k.streamId = 149 AND (k.end - k.start) > 20000000 -- > 20ms 的大内核 (主要是 Prefill / ViT)
ORDER BY k.start ASC;
"""
prefill_rows = cur.execute(query_prefills).fetchall()
print(f"Detected {len(prefill_rows)} major VLM inference sessions:")
for idx, (s, e, d) in enumerate(prefill_rows):
    print(f"  Session {idx+1}: start={s}, end={e}, dur={d:.1f}ms")

# 查看 Stream 149 上所有内核按时间排序的切分
# 找出 4 次测试 (0 FPS, 10 FPS, 22 FPS, 35 FPS)
# 每次测试之间有约 1 秒的 time.sleep(1.0)
query_vlm_kernels = """
SELECT k.start, k.end, (k.end - k.start) / 1e6 as dur_ms,
       (k.start - r.end) / 1000.0 as queue_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
WHERE k.streamId = 149
ORDER BY k.start ASC;
"""
vlm_ks = cur.execute(query_vlm_kernels).fetchall()

# 按照相邻内核间距 > 500ms 分割 session
sessions = []
current_session = []
for k in vlm_ks:
    if not current_session:
        current_session.append(k)
    else:
        gap_ms = (k[0] - current_session[-1][1]) / 1e6
        if gap_ms > 500.0:
            sessions.append(current_session)
            current_session = [k]
        else:
            current_session.append(k)
if current_session:
    sessions.append(current_session)

print(f"\nGrouped VLM kernels into {len(sessions)} distinct generation runs (sessions):")
fps_labels = ["热身阶段 (Warmup)", "1. 独占基准 (0 FPS)", "2. 并发阶段 (10 FPS)", "3. 并发阶段 (22 FPS)", "4. 并发阶段 (35 FPS)"]

for idx, s in enumerate(sessions):
    label = fps_labels[idx] if idx < len(fps_labels) else f"Session {idx+1}"
    t_start = s[0][0]
    t_end = s[-1][1]
    wall_ms = (t_end - t_start) / 1e6
    k_count = len(s)
    total_gpu_ms = sum(k[2] for k in s)
    q_lats = [k[3] for k in s if k[3] is not None and k[3] >= 0]
    avg_q_us = statistics.mean(q_lats) if q_lats else 0
    med_q_us = statistics.median(q_lats) if q_lats else 0
    p95_q_us = sorted(q_lats)[int(len(q_lats)*0.95)] if q_lats else 0

    # 同时统计该时间段内 Stream 17 (YOLO) 的活动情况
    yolo_active = cur.execute("""
        SELECT count(*), sum(end - start) / 1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE streamId = 17 AND start >= ? AND end <= ?
    """, (t_start, t_end)).fetchone()
    yolo_k_cnt = yolo_active[0] or 0
    yolo_gpu_ms = yolo_active[1] or 0.0

    print(f"\n--- [{label}] ---")
    print(f"  VLM 周期耗时     : {wall_ms:.1f} ms | 内核总数: {k_count} | 纯 GPU 计算时间: {total_gpu_ms:.1f} ms")
    print(f"  VLM 排队等待延迟 : 中位数={med_q_us:6.1f} us ({med_q_us/1000.0:.2f} ms) | 均值={avg_q_us:6.1f} us ({avg_q_us/1000.0:.2f} ms) | P95={p95_q_us:6.1f} us ({p95_q_us/1000.0:.2f} ms)")
    print(f"  同期 YOLO 活动   : 执行了 {yolo_k_cnt} 个卷积内核 | 霸占 GPU 累计: {yolo_gpu_ms:.1f} ms")
