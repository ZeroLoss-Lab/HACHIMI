#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provider Health Checker - 检测并移除持续失败的 provider

使用方法:
1. 检查所有 provider 健康状态:
   python provider_health_check.py

2. 自动移除失败的 provider (备份原文件):
   python provider_health_check.py --remove-failed

3. 手动指定阈值:
   python provider_health_check.py --remove-failed --failure-threshold 10
"""

import os, json, sys, argparse
from collections import defaultdict

FAILURES_FILE = "output/failures.jsonl"
KEYS_FILE = "secrets/key.txt"


def load_failures():
    """加载 failures.jsonl，统计每个 provider 的失败次数"""
    if not os.path.exists(FAILURES_FILE):
        print(f"[WARN] 未找到 {FAILURES_FILE}")
        return {}

    provider_failures = defaultdict(int)

    with open(FAILURES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                # failures.jsonl 格式: {"sid": 123, "error": "...", "client": "...", "ts": ...}
                error = entry.get("error", "")

                # 通过 error 内容识别 provider
                # 例如: "Provider QWEN-2 failed: HTTP 429"
                for provider_name in ["QWEN-2", "QWEN-3", "QWEN-4", "QWEN-5",
                                      "QWEN-6", "QWEN-7", "QWEN-8", "QWEN-9",
                                      "QWEN-10", "QWEN-11", "QWEN-12", "QWEN-13"]:
                    if provider_name.lower() in error.lower():
                        provider_failures[provider_name] += 1

            except Exception as e:
                print(f"[WARN] 解析失败行: {e}")
                continue

    return dict(provider_failures)


def load_key_file():
    """加载 key.txt 内容"""
    if not os.path.exists(KEYS_FILE):
        print(f"[ERROR] 未找到 {KEYS_FILE}")
        return []

    providers = []
    with open(KEYS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                obj = json.loads(line)
                providers.append(obj)
            except:
                # 尝试管道格式
                parts = line.split('|')
                if len(parts) >= 4:
                    providers.append({
                        "name": parts[0],
                        "base_url": parts[1],
                        "api_key": parts[2],
                        "model": parts[3]
                    })
    return providers


def backup_keys_file():
    """备份 key.txt"""
    if not os.path.exists(KEYS_FILE):
        return False

    import shutil
    backup_path = KEYS_FILE + ".backup"
    if os.path.exists(backup_path):
        os.remove(backup_path)
    shutil.copy2(KEYS_FILE, backup_path)
    print(f"[INFO] 已备份到 {backup_path}")
    return True


def save_key_file(providers):
    """保存更新后的 key.txt"""
    with open(KEYS_FILE, 'w', encoding='utf-8') as f:
        for p in providers:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"[INFO] 已更新 {KEYS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Provider Health Checker")
    parser.add_argument('--remove-failed', action='store_true',
                        help='自动移除失败的 provider')
    parser.add_argument('--failure-threshold', type=int, default=5,
                        help='失败阈值，超过此值认为 provider 不可用 (default: 5)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅显示哪些 provider 会被移除，不实际删除')

    args = parser.parse_args()

    print("="*60)
    print("Provider Health Checker")
    print("="*60)

    # 加载数据
    failures = load_failures()
    providers = load_key_file()

    if not providers:
        print(f"[ERROR] 无法加载 {KEYS_FILE}")
        return

    print(f"\n当前配置: {len(providers)} 个 providers")
    print(f"failures.jsonl 统计: {len(failures)} 个 provider 有失败记录")

    # 显示失败统计
    if failures:
        print("\n失败统计:")
        for provider, count in sorted(failures.items(), key=lambda x: x[1], reverse=True):
            status = "❌ 失败" if count >= args.failure_threshold else "⚠️ 警告"
            print(f"  {provider}: {count} 次失败 - {status}")

    # 识别应该移除的 provider
    to_remove = [name for name, count in failures.items()
                 if count >= args.failure_threshold]

    if not to_remove:
        print(f"\n✅ 所有 provider 都正常 (失败次数 < {args.failure_threshold})")
        return

    print(f"\n建议移除 ({len(to_remove)} 个):")
    for name in to_remove:
        print(f"  - {name}")

    # 移除逻辑
    if args.remove_failed or args.dry_run:
        if args.dry_run:
            print("\n[DRY RUN] 模拟移除操作:")
        else:
            print("\n开始移除失败的 provider...")
            backup_keys_file()

        # 过滤掉失败的 provider
        original_count = len(providers)
        providers_filtered = [p for p in providers if p.get("name") not in to_remove]
        removed_count = original_count - len(providers_filtered)

        print(f"移除 {removed_count} 个 provider 后剩余: {len(providers_filtered)} 个")

        if args.dry_run:
            print("[DRY RUN] 操作模拟完成，实际文件未修改")
            return

        if args.remove_failed:
            save_key_file(providers_filtered)
            print("\n✅ 更新完成！建议重新运行 CLI 任务。")
            print(f"\n注意: 原文件已备份到 {KEYS_FILE}.backup")
    else:
        print(f"\n提示: 使用 --remove-failed 参数自动移除这些 provider")
        print(f"或使用 --dry-run 查看哪些会被移除而不实际删除")


if __name__ == "__main__":
    main()