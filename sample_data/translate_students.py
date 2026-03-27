#!/usr/bin/env python3
"""
翻译学生数据文件：将所有字段名和值翻译成英语，并删除代理名字段
"""

import json
import requests
from tqdm import tqdm

# LLM API配置
API_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/60930ee7-02c0-4a60-bb43-aa934fbb5b65/f8d9c831-a053-4e2a-a33f-4aa90b2e69f2/proxy/9003/v1/chat/completions"
MODEL = "qwen"
API_KEY = "123"

# 字段名映射（中文 -> 英文）
FIELD_MAP = {
    "id": "id",
    "年龄": "age",
    "性别": "gender",
    "年级": "grade",
    "发展阶段": "development_stage",
    "代理名": "agent_name",  # 这个字段会被删除
    "擅长科目": "strong_subjects",
    "薄弱科目": "weak_subjects",
    "学术水平": "academic_level",
    "人格": "personality",
    "价值观": "values",
    "社交关系": "social_relationships",
    "创造力": "creativity",
    "心理健康": "mental_health",
    "_采样约束": "sampling_constraints"
}

# 发展阶段的子字段映射
DEV_STAGE_MAP = {
    "皮亚杰认知发展阶段": "piaget_cognitive_stage",
    "埃里克森心理社会发展阶段": "erikson_psychosocial_stage",
    "科尔伯格道德发展阶段": "kohlberg_moral_stage"
}

# 采样约束的子字段映射
CONSTRAINTS_MAP = {
    "年级": "grade",
    "性别": "gender",
    "优势学科偏向": "strength_subject_bias",
    "目标学术水平": "target_academic_level"
}

def call_llm(text, context=""):
    """调用本地Qwen模型进行翻译"""
    prompt = f"""Translate the following Chinese text to English. 
{context}

Chinese: {text}

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
        "max_tokens": 2048
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def translate_with_cache(text, cache, context=""):
    """带缓存的翻译函数"""
    if text in cache:
        return cache[text]
    
    translated = call_llm(text, context)
    if translated:
        cache[text] = translated
    return translated

def process_record(record, cache):
    """处理单条记录"""
    new_record = {}
    
    for key, value in record.items():
        # 跳过代理名字段
        if key == "代理名":
            continue
        
        # 获取英文字段名
        eng_key = FIELD_MAP.get(key, key)
        
        # 根据字段类型处理
        if key == "发展阶段":
            # 处理发展阶段的嵌套对象
            new_record[eng_key] = {}
            for sub_key, sub_value in value.items():
                eng_sub_key = DEV_STAGE_MAP.get(sub_key, sub_key)
                # 翻译值
                eng_sub_value = translate_with_cache(
                    sub_value, 
                    cache, 
                    f"This is a psychological development stage term."
                )
                new_record[eng_key][eng_sub_key] = eng_sub_value or sub_value
                
        elif key == "_采样约束":
            # 处理采样约束的嵌套对象
            new_record[eng_key] = {}
            for sub_key, sub_value in value.items():
                eng_sub_key = CONSTRAINTS_MAP.get(sub_key, sub_key)
                if isinstance(sub_value, list):
                    # 翻译列表中的每个元素
                    eng_list = []
                    for item in sub_value:
                        eng_item = translate_with_cache(
                            item, 
                            cache,
                            f"This is a subject or academic level term."
                        )
                        eng_list.append(eng_item or item)
                    new_record[eng_key][eng_sub_key] = eng_list
                else:
                    # 翻译字符串值
                    eng_sub_value = translate_with_cache(
                        sub_value, 
                        cache,
                        f"This is a grade level or gender term."
                    )
                    new_record[eng_key][eng_sub_key] = eng_sub_value or sub_value
                    
        elif key in ["擅长科目", "薄弱科目"]:
            # 处理科目列表
            eng_list = []
            for item in value:
                eng_item = translate_with_cache(
                    item, 
                    cache,
                    f"This is a school subject name."
                )
                eng_list.append(eng_item or item)
            new_record[eng_key] = eng_list
            
        elif key in ["id"]:
            # 保持id不变
            new_record[eng_key] = value
            
        elif key == "性别":
            # 翻译性别
            if value == "男":
                new_record[eng_key] = "male"
            elif value == "女":
                new_record[eng_key] = "female"
            else:
                new_record[eng_key] = value
                
        else:
            # 其他长文本字段需要翻译
            if isinstance(value, str):
                eng_value = translate_with_cache(
                    value, 
                    cache,
                    f"This is a student's {eng_key} description."
                )
                new_record[eng_key] = eng_value or value
            else:
                new_record[eng_key] = value
    
    return new_record

def main():
    input_file = "github-release-227/sample_data/merged_students_10k.jsonl"
    output_file = "github-release-227/sample_data/merged_students_10k_EN.jsonl"
    
    # 翻译缓存
    cache = {}
    
    # 先统计行数
    print("Counting total lines...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total records to process: {total_lines}")
    
    # 处理文件
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for i, line in enumerate(tqdm(fin, total=total_lines, desc="Translating")):
            try:
                record = json.loads(line.strip())
                translated_record = process_record(record, cache)
                fout.write(json.dumps(translated_record, ensure_ascii=False) + '\n')
                
                # 每10条保存一次缓存（可选）
                if (i + 1) % 10 == 0:
                    fout.flush()
                    
            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                continue
    
    print(f"\nTranslation complete!")
    print(f"Output file: {output_file}")
    print(f"Total cached translations: {len(cache)}")

if __name__ == "__main__":
    main()
