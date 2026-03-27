#!/usr/bin/env python3
"""
分块翻译长文本：支持断点续传，每100条保存一次
"""

import json
import requests
import concurrent.futures
from tqdm import tqdm
import threading
import os

translation_cache = {}
cache_lock = threading.Lock()

API_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/60930ee7-02c0-4a60-bb43-aa934fbb5b65/f8d9c831-a053-4e2a-a33f-4aa90b2e69f2/proxy/9003/v1/chat/completions"
MODEL = "qwen"
API_KEY = "123"
MAX_WORKERS = 50  # 降低并发提高稳定性
CHUNK_SIZE = 100  # 每100条保存一次

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
    input_file = "github-release-227/sample_data/merged_students_10k_EN.jsonl"
    temp_file = "github-release-227/sample_data/merged_students_10k_EN_partial.jsonl"
    
    # 如果存在部分完成的文件，从那里继续
    start_idx = 0
    records = []
    
    if os.path.exists(temp_file):
        print("Found partial file, resuming...")
        with open(temp_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
        start_idx = len(records)
        print(f"Resuming from record {start_idx}")
    
    # 读取剩余记录
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    if start_idx == 0:
        records = [json.loads(line) for line in all_lines]
    else:
        # 验证已处理的部分
        print(f"Already have {len(records)} translated records")
        remaining = [json.loads(line) for line in all_lines[start_idx:]]
        print(f"Remaining to translate: {len(remaining)}")
        
        # 分块处理剩余部分
        for chunk_start in range(0, len(remaining), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(remaining))
            chunk = remaining[chunk_start:chunk_end]
            
            print(f"\nProcessing chunk {chunk_start//CHUNK_SIZE + 1}: records {start_idx + chunk_start} to {start_idx + chunk_end}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(tqdm(
                    executor.map(process_record, chunk),
                    total=len(chunk),
                    desc=f"Chunk {chunk_start//CHUNK_SIZE + 1}"
                ))
            
            records.extend(results)
            
            # 保存进度
            with open(temp_file, 'w', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"Saved progress: {len(records)} records")
    
    # 如果没有部分文件，从头开始
    if start_idx == 0:
        for chunk_start in range(0, len(records), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(records))
            chunk = records[chunk_start:chunk_end]
            
            print(f"\nProcessing chunk {chunk_start//CHUNK_SIZE + 1}: records {chunk_start} to {chunk_end}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                translated_chunk = list(tqdm(
                    executor.map(process_record, chunk),
                    total=len(chunk),
                    desc=f"Chunk {chunk_start//CHUNK_SIZE + 1}"
                ))
            
            # 更新records
            for i, r in enumerate(translated_chunk):
                records[chunk_start + i] = r
            
            # 保存进度
            with open(temp_file, 'w', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"Saved progress: {chunk_end}/{len(records)} records")
    
    # 完成，重命名文件
    os.rename(temp_file, input_file)
    print(f"Done! Total records: {len(records)}")
    print(f"Cache size: {len(translation_cache)}")

if __name__ == "__main__":
    main()
