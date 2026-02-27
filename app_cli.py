import os
os.environ["HACHIMI_CLI"] = "1"

import sys
sys.argv.append("--cli")

# 延迟导入以加速启动
def _import_modules():
    from app_fixing import (
        Orchestrator, SimilarityGate, QuotaScheduler,
        _resolve_run_dir, _load_meta, _save_meta, _load_schedule, _save_schedule,
        _scan_completed_ids, _append_failure, _count_lines, _worker_process_one,
        _estimate_concurrency, _parse_level_mix,
    )
    from providers import load_providers, ProviderPool
    import json, glob, time, threading
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed
    return (Orchestrator, SimilarityGate, QuotaScheduler,
            _resolve_run_dir, _load_meta, _save_meta, _load_schedule, _save_schedule,
            _scan_completed_ids, _append_failure, _count_lines, _worker_process_one,
            _estimate_concurrency, _parse_level_mix,
            load_providers, ProviderPool,
            json, glob, time, threading, defaultdict, ThreadPoolExecutor, as_completed)

(Orchestrator, SimilarityGate, QuotaScheduler,
 _resolve_run_dir, _load_meta, _save_meta, _load_schedule, _save_schedule,
 _scan_completed_ids, _append_failure, _count_lines, _worker_process_one,
 _estimate_concurrency, _parse_level_mix,
 load_providers, ProviderPool,
 json, glob, time, threading, defaultdict, ThreadPoolExecutor, as_completed) = _import_modules()

# ===== 用户可修改的配置 =====
RUN_ID = "72b_50k"          # 输出目录名，结果在 ./output/{RUN_ID}/
TOTAL = 50000                  # 生成总数，先小量验证再调大
CHUNK_SIZE = 50              # 每个 jsonl 片的行数
MAX_ROUNDS = 3               # Validator-deep 轮数
SIMHASH = 3                  # SimHash Hamming 阈值
KEYS_PATH = os.path.join(os.path.dirname(__file__), "secrets", "providers.json")
MAX_WORKERS = None           # 线程数；None 表示自动估计
LEVEL_MIX = None             # 学术水平配比，如 "高0.25,中0.25,低0.25,差0.25"
QUOTA_JSON = None            # 自定义配额 JSON（优先于 LEVEL_MIX）
PER_STEP_M = 20              # 每步并发生成条数（服务器能承受 1500）
RETRY_LIMIT = 4              # 单 sid 最多重试次数
# ============================

def _key_text_for_simhash(item: dict) -> str:
    return "|".join([
        str(item.get("年级", "")),
        str(item.get("性别", "")),
        str(item.get("人格", "")),
        str(item.get("价值观", "")),
        str(item.get("社交关系", "")),
        str(item.get("创造力", "")),
        str(item.get("心理健康", "")),
    ])

def _seed_sim_gate_from_disk(run_dir: str, sim_gate: SimilarityGate):
    """种子化相似性门控（避免生成重复）"""
    import glob
    files = glob.glob(os.path.join(run_dir, "students_chunk_*.jsonl"))
    if not files:
        print(f"    [seeding] No files found in {run_dir}")
        return

    t0 = time.time()
    total = 0
    print(f"    [seeding] 预估 ~730k 记录，需要 ~10-20秒...")

    # 分批读取以显示进度
    batch_size = min(50, len(files))
    for batch_start in range(0, len(files), batch_size):
        batch_end = min(batch_start + batch_size, len(files))
        batch_t0 = time.time()

        for idx in range(batch_start, batch_end):
            p = files[idx]
            try:
                with open(p, "r", encoding="utf-8", errors="surrogateescape") as f:
                    for ln in f:
                        if not ln.strip():
                            continue
                        try:
                            obj = json.loads(ln.strip())
                            sim_gate.accept(_key_text_for_simhash(obj))
                            total += 1
                        except Exception:
                            pass
            except Exception:
                continue

        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"    [seeding] {batch_end}/{len(files)} files, {total} records ({rate:.0f}/s)")

    elapsed = time.time() - t0
    print(f"    [seeding] DONE: {total} records from {len(files)} files in {elapsed:.1f}s")

def _append_record_cli(run_dir: str, chunk_size: int, record: dict,
                       completed: set, write_lock: threading.Lock) -> bool:
    sid = int(record.get("id", 0))
    if sid <= 0:
        raise RuntimeError("record 缺少合法 id")
    chunk_no = (sid - 1) // chunk_size + 1
    path = os.path.join(run_dir, f"students_chunk_{chunk_no}.jsonl")
    line = json.dumps(record, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with write_lock:
        if sid in completed:
            return False
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        completed.add(sid)
        return True

def run_cli_batch(run_id: str, total_n: int, chunk_size: int, max_rounds: int,
                  simhash_threshold: int, keys_path: str,
                  max_workers: int | None = None,
                  level_mix_text: str | None = None,
                  quota_json: str | None = None,
                  per_step_m: int = 10,
                  retry_limit: int = 4):
    run_dir = _resolve_run_dir(run_id)
    os.makedirs(run_dir, exist_ok=True)

    meta = _load_meta(run_dir)
    if meta:
        if int(meta.get("total_n", total_n)) != int(total_n) or int(meta.get("chunk_size", chunk_size)) != int(chunk_size):
            raise ValueError(f"meta.json 与当前参数不一致: meta={meta}, 请求 total_n={total_n}, chunk_size={chunk_size}")
    else:
        _save_meta(run_dir, total_n, chunk_size)

    level_mix = _parse_level_mix(level_mix_text) if level_mix_text else None

    print(f"[CLI] 加载/生成调度表...")
    t0 = time.time()

    # 先扫描已完成列表
    print(f"  -> 扫描已完成记录...")
    completed = _scan_completed_ids(run_dir)
    pending = [sid for sid in range(1, total_n + 1) if sid not in completed]
    print(f"  -> 扫描耗时: {time.time()-t0:.2f}s (已完成 {len(completed)}，待完成 {len(pending)})")

    print(f"  -> 加载调度表...")
    schedule = _load_schedule(run_dir)
    if schedule is None:
        print(f"  -> 磁盘无 schedule.json，生成中...")
        qs = QuotaScheduler(total_n, user_quota_json=quota_json, level_mix=level_mix)
        schedule = qs.slots
        print(f"  -> 保存到磁盘 ({len(schedule)} 条)...")
        _save_schedule(run_dir, schedule)
        print(f"  -> 耗时: {time.time()-t0:.2f}s")
    else:
        if len(pending) == 0:
            print(f"  -> 跳过 schedule 加载（所有任务已完成）")
        else:
            print(f"  -> 从磁盘加载 ({len(schedule)} 条)")
        print(f"  -> 耗时: {time.time()-t0:.2f}s")

    if len(schedule) < total_n:
        raise ValueError(f"schedule 长度不足 total_n: len={len(schedule)} < {total_n}")

    print(f"  -> 初始化相似性过滤...")
    # completed/pending 已在前面扫描过
    sim_gate = SimilarityGate(threshold=simhash_threshold)
    _seed_sim_gate_from_disk(run_dir, sim_gate)

    print(f"  -> 加载 Provider 池...")
    pool = ProviderPool(load_providers(keys_path))
    orch = Orchestrator(max_rounds=max_rounds)
    write_lock = threading.Lock()
    attempts = defaultdict(int)

    print(f"[CLI] run_dir={run_dir}, total_n={total_n}, chunk_size={chunk_size}, pending={len(pending)}, completed={len(completed)}")

    if len(pending) == 0:
        print(f"\n[CLI] ✓ 所有任务已完成！")
        print(f"如需生成新数据，请修改 RUN_ID 或删除/移动当前输出目录。")
        return

    batch_id = 0
    total_generated = 0
    t_start = time.time()

    while pending:
        batch_id += 1
        C_star = _estimate_concurrency(pool, cmax=max_workers or 1024)
        batch_base = max(int(per_step_m), int(2 * max(1, C_star)))
        batch_size = max(1, min(len(pending), batch_base))
        scheduled = [pending.pop(0) for _ in range(batch_size)]
        max_workers_now = max(1, min(len(scheduled), C_star if C_star > 0 else len(scheduled)))

        print(f"[CLI][batch {batch_id}] 调度 {batch_size} 条, 并发 {max_workers_now}, 待完成 {len(pending)}")
        print(f"[CLI][batch {batch_id}] 开始处理... (已提交到线程池)")

        futures = {}
        batch_start = time.time()

        # 提交所有任务到线程池 (非阻塞)
        with ThreadPoolExecutor(max_workers=max_workers_now) as ex:
            for sid in scheduled:
                client = pool.pick()
                slot = schedule[sid - 1] if sid - 1 < len(schedule) else {}
                fut = ex.submit(_worker_process_one, orch, sid, slot, client, sim_gate, retry_limit)
                futures[fut] = sid

            # 实时处理完成的任务 (模拟前端 behavior)
            completed_in_batch = 0
            failed_in_batch = 0

            for fut in as_completed(futures):
                sid = futures[fut]
                ok, item, dialog, err = fut.result()

                if ok and item:
                    wrote = _append_record_cli(run_dir, chunk_size, item, completed, write_lock)
                    if wrote:
                        total_generated += 1
                        completed_in_batch += 1
                        # 实时输出（不等整个 batch）
                        elapsed = time.time() - t_start
                        rate = total_generated / elapsed if elapsed > 0 else 0
                        print(f"[CLI][batch {batch_id}] ✓ sid={sid} 已落盘 | 总进度: {total_generated}/{total_n} ({rate:.2f}/s)")
                else:
                    attempts[sid] += 1
                    failed_in_batch += 1
                    if "SimHash" in (err or "") or attempts[sid] >= retry_limit:
                        _append_failure(run_dir, {"sid": sid, "error": err, "attempts": attempts[sid], "ts": time.time()})
                        print(f"[CLI][batch {batch_id}] ✗ sid={sid} 失败(不再重试): {err}")
                    else:
                        pending.append(sid)
                        print(f"[CLI][batch {batch_id}] ↻ sid={sid} 刷回，将重试: {err}")

        batch_elapsed = time.time() - batch_start
        print(f"[CLI][batch {batch_id}] 完成: {completed_in_batch} 成功, {failed_in_batch} 失败, 耗时 {batch_elapsed:.1f}s")

    total_time = time.time() - t_start
    print(f"[CLI] 完成: 共生成 {total_generated} 条，总耗时 {total_time:.1f}s ({total_generated/total_time:.2f}/s)")
    print(f"跳过失败 {len([k for k,v in attempts.items() if v>=retry_limit])} 条")

def main():
    run_cli_batch(
        run_id=RUN_ID,
        total_n=TOTAL,
        chunk_size=CHUNK_SIZE,
        max_rounds=MAX_ROUNDS,
        simhash_threshold=SIMHASH,
        keys_path=KEYS_PATH,
        max_workers=MAX_WORKERS,
        level_mix_text=LEVEL_MIX,
        quota_json=QUOTA_JSON,
        per_step_m=PER_STEP_M,
        retry_limit=RETRY_LIMIT,
    )

if __name__ == "__main__":
    main()