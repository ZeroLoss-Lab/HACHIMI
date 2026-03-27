#!/usr/bin/env python3
"""
翻译长描述性文本字段：使用100并发LLM翻译
"""

import json
import requests
import concurrent.futures
from tqdm import tqdm
import threading

# ==================== 配置 ====================
API_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/60930ee7-02c0-4a60-bb43-aa934fbb5b65/f8d9c831-a053-4e2a-a33f-4aa90b2e69f2/proxy/9003/v1/chat/completions"
MODEL = "qwen"
API_KEY = "123"
MAX_WORKERS = 100  # 并发数

# 需要翻译的长文本字段
TEXT_FIELDS = ['personality', 'values', 'social_relationships', 'creativity', 'mental_health']

# 翻译缓存
translation_cache = {}
cache_lock = threading.Lock()

def call_llm(text, field_name):
    """调用本地Qwen模型进行翻译"""
    if not text or len(text.strip()) == 0:
        return text
    
    # 检查缓存
    with cache_lock:
        if text in translation_cache:
            return translation_cache[text]
    
    prompt = f"""Translate the following Chinese text to English. This is a student's {field_name} description in educational psychology context. Keep the meaning accurate and natural.

Chinese:
{text}

English:"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4096
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()
        translated = result["choices"][0]["message"]["content"].strip()
        
        # 保存到缓存
        with cache_lock:
            translation_cache[text] = translated
        
        return translated
    except Exception as e:
        print(f"Error translating {field_name}: {e}")
        return None

def process_record(record_with_index):
    """处理单条记录"""
    idx, record = record_with_index
    
    for field in TEXT_FIELDS:
        if field in record and record[field]:
            translated = call_llm(record[field], field)
            if translated:
                record[field] = translated
    
    return idx, record

def main():
    input_file = "github-release-227/sample_data/merged_students_10k_EN.jsonl"
    output_file = "github-release-227/sample_data/merged_students_10k_EN.jsonl"
    
    # 读取所有记录
    print("Loading records...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Total records to process: {len(records)}")
    print(f"Concurrency: {MAX_WORKERS}")
    print(f"Fields to translate: {TEXT_FIELDS}")
    
    # 并发处理
    print(f"Processing with {MAX_WORKERS} workers...")
    translated_records = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_record, (i, record)): i 
                   for i, record in enumerate(records)}
        
        with tqdm(total=len(records), desc="Translating") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, translated_record = future.result()
                    translated_records.append((idx, translated_record))
                except Exception as e:
                    print(f"Error processing record: {e}")
                pbar.update(1)
    
    # 按原始顺序排序
    translated_records.sort(key=lambda x: x[0])
    
    # 写入输出文件
    print("Writing output file...")
    with open(output_file, 'w', encoding='utf-8') as fout:
        for _, record in translated_records:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\nTranslation complete!")
    print(f"Output file: {output_file}")
    print(f"Total cached translations: {len(translation_cache)}")

if __name__ == "__main__":
    main()
