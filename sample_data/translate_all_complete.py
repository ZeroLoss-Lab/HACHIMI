#!/usr/bin/env python3
"""
完整翻译：从原始文件开始，翻译所有字段（包括长文本），100并发
"""

import json
import requests
import concurrent.futures
from tqdm import tqdm
import threading
import re

# ==================== 配置 ====================
API_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/60930ee7-02c0-4a60-bb43-aa934fbb5b65/f8d9c831-a053-4e2a-a33f-4aa90b2e69f2/proxy/9003/v1/chat/completions"
MODEL = "qwen"
API_KEY = "123"
MAX_WORKERS = 100

# ==================== 映射表 ====================
GENDER_MAP = {"男": "male", "女": "female"}

GRADE_MAP = {
    "一年级": "Grade 1 (Primary)", "二年级": "Grade 2 (Primary)",
    "三年级": "Grade 3 (Primary)", "四年级": "Grade 4 (Primary)",
    "五年级": "Grade 5 (Primary)", "六年级": "Grade 6 (Primary)",
    "初一": "Grade 7 (Junior High)", "初二": "Grade 8 (Junior High)", "初三": "Grade 9 (Junior High)",
    "高一": "Grade 10 (Senior High)", "高二": "Grade 11 (Senior High)", "高三": "Grade 12 (Senior High)",
}

SUBJECT_MAP = {
    "语文": "Chinese Language", "数学": "Mathematics", "英语": "English",
    "物理": "Physics", "化学": "Chemistry", "生物": "Biology",
    "历史": "History", "地理": "Geography", "政治": "Politics",
    "道德与法治": "Morality and Law", "美术": "Art", "音乐": "Music",
    "体育": "Physical Education", "信息技术": "Information Technology",
    "科学": "Science", "书法": "Calligraphy", "综合实践": "Comprehensive Practice",
    "劳动与技术": "Labor and Technology", "思想品德": "Moral Education",
}

ACADEMIC_LEVEL_MAP = {
    "差：成绩全校排名后50%": "Poor: Bottom 50% of school",
    "低：成绩全校排名前30%至50%": "Low: Top 30%-50% of school",
    "中：成绩全校排名前10%至30%": "Average: Top 10%-30% of school",
    "高：成绩全校排名前10%": "High: Top 10% of school",
}

def smart_translate_piaget(text):
    if not text: return text
    if "形式运算" in text: return "Formal Operational Stage"
    if "具体运算" in text: return "Concrete Operational Stage"
    if "前运算" in text: return "Preoperational Stage"
    return "Formal Operational Stage"

def smart_translate_erikson(text):
    if not text: return text
    if "勤奋" in text or "自卑" in text: return "Industry vs. Inferiority"
    if "主动" in text or "内疚" in text: return "Initiative vs. Guilt"
    if "信任" in text: return "Trust vs. Mistrust"
    if "自主" in text: return "Autonomy vs. Shame"
    if "亲密" in text: return "Intimacy vs. Isolation"
    return "Identity vs. Role Confusion"

def smart_translate_kohlberg(text):
    if not text: return text
    if "前" in text: return "Preconventional Level"
    if "后" in text or "原则" in text: return "Postconventional Level"
    return "Conventional Level"

# ==================== LLM翻译 ====================
translation_cache = {}
cache_lock = threading.Lock()

def call_llm(text, field_name):
    if not text or len(text.strip()) < 10:
        return text
    
    with cache_lock:
        if text in translation_cache:
            return translation_cache[text]
    
    prompt = f"""Translate the following Chinese text to English. This is a student's {field_name} description in educational psychology context. Translate accurately and naturally.

Chinese:
{text}

English:"""
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 4096}
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()
        translated = result["choices"][0]["message"]["content"].strip()
        with cache_lock:
            translation_cache[text] = translated
        return translated
    except Exception as e:
        print(f"Error: {e}")
        return None

# ==================== 处理记录 ====================

def process_record(record_with_index):
    idx, record = record_with_index
    new_record = {}
    
    for key, value in record.items():
        if key == "代理名":
            continue
        
        # 字段名映射
        field_map = {
            "id": "id", "年龄": "age", "性别": "gender", "年级": "grade",
            "发展阶段": "development_stage", "擅长科目": "strong_subjects",
            "薄弱科目": "weak_subjects", "学术水平": "academic_level",
            "人格": "personality", "价值观": "values",
            "社交关系": "social_relationships", "创造力": "creativity",
            "心理健康": "mental_health", "_采样约束": "sampling_constraints"
        }
        eng_key = field_map.get(key, key)
        
        if key == "发展阶段":
            new_record[eng_key] = {
                "piaget_cognitive_stage": smart_translate_piaget(value.get("皮亚杰认知发展阶段", "")),
                "erikson_psychosocial_stage": smart_translate_erikson(value.get("埃里克森心理社会发展阶段", "")),
                "kohlberg_moral_stage": smart_translate_kohlberg(value.get("科尔伯格道德发展阶段", ""))
            }
        elif key == "_采样约束":
            new_record[eng_key] = {
                "grade": GRADE_MAP.get(value.get("年级", ""), value.get("年级", "")),
                "gender": GENDER_MAP.get(value.get("性别", ""), value.get("性别", "")),
                "strength_subject_bias": [SUBJECT_MAP.get(s, s) for s in value.get("优势学科偏向", [])],
                "target_academic_level": ACADEMIC_LEVEL_MAP.get(value.get("目标学术水平", ""), value.get("目标学术水平", ""))
            }
        elif key in ["擅长科目", "薄弱科目"]:
            new_record[eng_key] = [SUBJECT_MAP.get(s, s) for s in value]
        elif key == "id":
            new_record[eng_key] = value
        elif key == "性别":
            new_record[eng_key] = GENDER_MAP.get(value, value)
        elif key == "年级":
            new_record[eng_key] = GRADE_MAP.get(value, value)
        elif key == "学术水平":
            new_record[eng_key] = ACADEMIC_LEVEL_MAP.get(value, value)
        elif key in ["人格", "价值观", "社交关系", "创造力", "心理健康"]:
            # 使用LLM翻译长文本
            field_names = {"人格": "personality", "价值观": "values", "社交关系": "social relationships", "创造力": "creativity", "心理健康": "mental health"}
            translated = call_llm(value, field_names.get(key, key))
            new_record[eng_key] = translated if translated else value
        else:
            new_record[eng_key] = value
    
    return idx, new_record

# ==================== 主函数 ====================

def main():
    input_file = "github-release-227/sample_data/merged_students_10k.jsonl"
    output_file = "github-release-227/sample_data/merged_students_10k_EN.jsonl"
    
    print("Loading records...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except:
                continue
    
    print(f"Total: {len(records)}, Concurrency: {MAX_WORKERS}")
    
    # 并发处理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_record, (i, r)): i for i, r in enumerate(records)}
        with tqdm(total=len(records), desc="Translating") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, record = future.result()
                    results.append((idx, record))
                except Exception as e:
                    print(f"Error: {e}")
                pbar.update(1)
    
    # 排序并写入
    results.sort(key=lambda x: x[0])
    with open(output_file, 'w', encoding='utf-8') as fout:
        for _, record in results:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Done! Cache size: {len(translation_cache)}")

if __name__ == "__main__":
    main()
