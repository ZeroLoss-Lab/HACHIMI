#!/usr/bin/env python3
"""
分批翻译：每次处理BATCH_SIZE条，支持断点续传
用法: python translate_batch.py [start_index] [batch_size]
"""

import json
import requests
import concurrent.futures
from tqdm import tqdm
import threading
import sys

translation_cache = {}
cache_lock = threading.Lock()

API_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/60930ee7-02c0-4a60-bb43-aa934fbb5b65/f8d9c831-a053-4e2a-a33f-4aa90b2e69f2/proxy/9003/v1/chat/completions"
MODEL = "qwen"
API_KEY = "123"
MAX_WORKERS = 50

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
        print(f"Error: {e}")
        return None

def process_record(record):
    for field in TEXT_FIELDS:
        if field in record and record[field]:
            translated = call_llm(record[field], FIELD_NAMES[field])
            if translated:
                record[field] = translated
    return record

def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    input_file = "github-release-227/sample_data/merged_students_10k_EN.jsonl"
    output_file = "github-release-227/sample_data/merged_students_10k_EN.jsonl"
    
    # 读取所有记录（忽略编码错误）
    records = []
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass
    
    end_idx = min(start_idx + batch_size, len(records))
    print(f"Processing records {start_idx} to {end_idx} (total: {len(records)})")
    
    batch = records[start_idx:end_idx]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        translated = list(tqdm(
            executor.map(process_record, batch),
            total=len(batch),
            desc="Translating"
        ))
    
    # 更新records
    for i, r in enumerate(translated):
        records[start_idx + i] = r
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\nDone! Processed {end_idx}/{len(records)} records")
    print(f"Cache size: {len(translation_cache)}")
    
    if end_idx < len(records):
        print(f"\nNext run: python translate_batch.py {end_idx} {batch_size}")

if __name__ == "__main__":
    main()
