import sqlite3
from collections import defaultdict
import statistics

db_path = r"C:\Users\chanceyu\Downloads\guard_contention_profile.sqlite"
con = sqlite3.connect(db_path)
cur = con.cursor()

# 1. 提取所有字符串映射
print("Loading StringIds...")
strings = dict(cur.execute("SELECT id, value FROM StringIds").fetchall())

# 2. 统计 CUDA Streams
print("\n=== CUDA STREAMS & KERNEL DISTRIBUTION ===")
query_streams = """
SELECT streamId, count(*) as k_count, 
       min(start) as min_start, max(end) as max_end,
       sum(end - start) / 1000000.0 as total_gpu_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY streamId
ORDER BY k_count DESC;
"""
print(f"{'StreamId':<10} | {'KernelCount':<12} | {'TotalGPUMs':<12} | {'TimeRange_ms'}")
print("-" * 65)
stream_rows = cur.execute(query_streams).fetchall()
for sid, cnt, min_s, max_e, tot_ms in stream_rows:
    time_span = (max_e - min_s) / 1e6
    print(f"{sid:<10} | {cnt:<12} | {tot_ms:<12.2f} | {time_span:.1f} ms")

# 3. 统计最耗时的 Top 30 内核及其所属 Stream
query_top_kernels = """
SELECT streamId, demangledName, count(*) as exec_count,
       sum(end - start) / 1000000.0 as total_duration_ms,
       avg(end - start) / 1000.0 as avg_duration_us
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY streamId, demangledName
ORDER BY total_duration_ms DESC
LIMIT 30;
"""
top_rows = cur.execute(query_top_kernels).fetchall()
print("\n=== TOP 30 KERNELS BY DURATION ===")
for sid, name_id, cnt, tot_ms, avg_us in top_rows:
    kname = strings.get(name_id, f"id_{name_id}")
    short_kname = kname[:85] if len(kname) > 85 else kname
    print(f"Stream {sid:2d} | Total: {tot_ms:8.2f}ms | Calls: {cnt:5d} | Avg: {avg_us:6.1f}us | {short_kname}")

# 4. 分离流特征与模型分类
query_all = """
SELECT start, end, streamId, demangledName, correlationId
FROM CUPTI_ACTIVITY_KIND_KERNEL
ORDER BY start ASC;
"""
print("\nClassifying kernels into VLM vs YOLO across timeline...")
cur.execute(query_all)
all_k = cur.fetchall()

vlm_streams = set()
yolo_streams = set()
kernel_records = []

for start, end, streamId, nameId, corrId in all_k:
    kname = strings.get(nameId, "").lower()
    is_vlm = any(kw in kname for kw in ['edgellm', 'causal_conv1d', 'mrope', 'decoding', 'kv_cache', 'rmsnorm', 'token', 'embedding', 'rotary'])
    is_yolo = any(kw in kname for kw in ['c3k2', 'conv', 'implicit_gemm', 'yolo', 'detect', 'upsample', 'maxpool'])
    
    cat = 'UNKNOWN'
    if is_vlm:
        cat = 'VLM'
        vlm_streams.add(streamId)
    elif is_yolo:
        cat = 'YOLO'
        yolo_streams.add(streamId)
    
    kernel_records.append((start, end, streamId, cat, kname))

print(f"Identified VLM Streams : {sorted(list(vlm_streams))}")
print(f"Identified YOLO Streams: {sorted(list(yolo_streams))}")

# 5. 分析时间重叠 (Concurrency vs Serialization)
cur_vlm_end = 0
cur_yolo_end = 0
total_vlm_time_ns = 0
total_yolo_time_ns = 0
total_overlap_ns = 0
vlm_yolo_overlap_count = 0

for start, end, sid, cat, kname in kernel_records:
    dur = end - start
    if cat == 'VLM':
        total_vlm_time_ns += dur
        if start < cur_yolo_end:
            overlap = min(end, cur_yolo_end) - start
            if overlap > 0:
                total_overlap_ns += overlap
                vlm_yolo_overlap_count += 1
        if end > cur_vlm_end:
            cur_vlm_end = end
    elif cat == 'YOLO':
        total_yolo_time_ns += dur
        if start < cur_vlm_end:
            overlap = min(end, cur_vlm_end) - start
            if overlap > 0:
                total_overlap_ns += overlap
                vlm_yolo_overlap_count += 1
        if end > cur_yolo_end:
            cur_yolo_end = end

print("\n=== CONCURRENCY & TEMPORAL OVERLAP ANALYSIS ===")
print(f"Total VLM GPU Active Time  : {total_vlm_time_ns / 1e6:10.2f} ms")
print(f"Total YOLO GPU Active Time : {total_yolo_time_ns / 1e6:10.2f} ms")
print(f"Total Overlapping GPU Time : {total_overlap_ns / 1e6:10.2f} ms")
print(f"Overlap Incident Count     : {vlm_yolo_overlap_count}")
if total_vlm_time_ns > 0:
    print(f"Overlap Ratio (Overlap / VLM Time): {total_overlap_ns / total_vlm_time_ns * 100.0:.2f}%")

# 6. 分析各个测试阶段的单次迭代表现 (按时间窗口切割)
# 找出 4 个测试阶段的时间区间
# 根据 NVTX 标记或大段间歇
gaps = []
for i in range(1, len(all_k)):
    gap_ms = (all_k[i][0] - all_k[i-1][1]) / 1e6
    if gap_ms > 200.0: # 超过200ms的间歇通常是测试轮次之间的切换
        gaps.append((all_k[i-1][1], all_k[i][0], gap_ms))

print(f"\nDetected {len(gaps)} major execution phase transitions (gaps > 200ms):")
for idx, (g_start, g_end, gap_ms) in enumerate(gaps):
    print(f"  Phase boundary {idx+1}: gap of {gap_ms:.1f} ms")

# 7. 排队延迟统计 (Queue Latency)
query_queue = """
SELECT k.streamId, 
       (k.start - r.end) / 1000.0 as queue_latency_us,
       (k.end - k.start) / 1000.0 as dur_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
WHERE (k.start - r.end) >= 0;
"""
queue_data = cur.execute(query_queue).fetchall()
print(f"\n=== CUDA LAUNCH-TO-EXECUTE QUEUE LATENCY ===")
print(f"Total analyzed launch-kernel pairs: {len(queue_data)}")

stream_latencies = defaultdict(list)
all_lats = []
for sid, q_us, dur_us in queue_data:
    stream_latencies[sid].append(q_us)
    all_lats.append(q_us)

if all_lats:
    all_lats.sort()
    n = len(all_lats)
    print(f"Global Queue Latency: Mean={statistics.mean(all_lats):.1f}us | Median={all_lats[n//2]:.1f}us | P95={all_lats[int(n*0.95)]:.1f}us | Max={all_lats[-1]:.1f}us")

print("\nQueue Latency by Stream:")
print(f"{'StreamId':<10} | {'Count':<8} | {'Mean(us)':<10} | {'Median(us)':<12} | {'P95(us)':<10} | {'Max(us)'}")
print("-" * 65)
for sid, lats in sorted(stream_latencies.items()):
    lats.sort()
    m = len(lats)
    print(f"{sid:<10} | {m:<8} | {statistics.mean(lats):<10.1f} | {lats[m//2]:<12.1f} | {lats[int(m*0.95)]:<10.1f} | {lats[-1]:.1f}")
