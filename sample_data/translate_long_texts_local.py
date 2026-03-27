#!/usr/bin/env python3
"""
本地断点续传翻译脚本 - 适合长时间运行
使用方法:
1. 直接运行: python translate_long_texts_local.py
2. 脚本会自动保存进度，中断后重新运行会从断点继续
3. 全部完成后会生成最终的 EN.jsonl 文件
"""

import json
import requests
import concurrent.futures
from tqdm import tqdm
import threading
import os
import sys

translation_cache = {}
cache_lock = threading.Lock()

# API配置
API_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/60930ee7-02c0-4a60-bb43-aa934fbb5b65/f8d9c831-a053-4e2a-a33f-4aa90b2e69f2/proxy/9003/v1/chat/completions"
MODEL = "qwen"
API_KEY = "123"
MAX_WORKERS = 80  # 并发数
BATCH_SIZE = 5    # 每5条保存一次，更快保存

TEXT_FIELDS = ['personality', 'values', 'social_relationships', 'creativity', 'mental_health']
FIELD_NAMES = {
    'personality': 'personality',
    'values': 'values',
    'social_relationships': 'social relationships',
    'creativity': 'creativity',
    'mental_health': 'mental health'
}

def call_llm(text, field_name):
    if not text or len(text.strip()) < 10:
        return text
    with cache_lock:
        if text in translation_cache:
            return translation_cache[text]
    
    prompt = f"""Translate the following Chinese text to English. This is a student's {field_name} description in educational psychology context.

Chinese:
{text}

English:"""
    
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 4096},
            timeout=180
        )
        response.raise_for_status()
        translated = response.json()["choices"][0]["message"]["content"].strip()
        with cache_lock:
            translation_cache[text] = translated
        return translated
    except Exception as e:
        print(f"\nError translating {field_name}: {e}")
        return None

def process_record(record):
    for field in TEXT_FIELDS:
        if field in record and record[field]:
            translated = call_llm(record[field], FIELD_NAMES[field])
            if translated:
                record[field] = translated
    return record

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "merged_students_10k_EN.jsonl")
    progress_file = os.path.join(script_dir, "translation_progress.json")
    
    # 读取所有记录
    print("Loading records...")
    records = []
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass
    
    total = len(records)
    print(f"Total records: {total}")
    
    # 检查进度
    start_idx = 0
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            start_idx = progress.get('last_processed', 0)
        print(f"Resuming from record {start_idx}")
    
    if start_idx >= total:
        print("All records already processed!")
        return
    
    # 分块处理
    for batch_start in range(start_idx, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = records[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"Processing batch: {batch_start} to {batch_end} ({batch_end}/{total})")
        print(f"{'='*60}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            translated = list(tqdm(
                executor.map(process_record, batch),
                total=len(batch),
                desc=f"Batch {batch_start//BATCH_SIZE + 1}"
            ))
        
        # 更新records
        for i, r in enumerate(translated):
            records[batch_start + i] = r
        
        # 保存进度
        with open(progress_file, 'w') as f:
            json.dump({'last_processed': batch_end}, f)
        
        # 保存中间结果
        with open(input_file, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        print(f"Saved progress: {batch_end}/{total} records")
        print(f"Cache size: {len(translation_cache)}")
    
    print("\n" + "="*60)
    print("ALL DONE! Translation complete!")
    print("="*60)
    
    # 清理进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress has been saved. Run again to resume.")
        sys.exit(0)
