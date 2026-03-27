#!/usr/bin/env python3
"""
翻译学生数据文件：使用统一标准映射表 + 100并发LLM翻译
"""

import json
import requests
import concurrent.futures
from tqdm import tqdm
import threading
import time

# ==================== 配置 ====================
API_URL = "https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-c6264b38-46a1-4bb6-a3a7-1db39453f6f8/vscode/60930ee7-02c0-4a60-bb43-aa934fbb5b65/f8d9c831-a053-4e2a-a33f-4aa90b2e69f2/proxy/9003/v1/chat/completions"
MODEL = "qwen"
API_KEY = "123"
MAX_WORKERS = 100  # 并发数

# ==================== 统一标准映射表 ====================

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

# 性别映射
GENDER_MAP = {
    "男": "male",
    "女": "female"
}

# 年级映射 - 统一标准
GRADE_MAP = {
    "一年级": "Grade 1 (Primary)",
    "二年级": "Grade 2 (Primary)",
    "三年级": "Grade 3 (Primary)",
    "四年级": "Grade 4 (Primary)",
    "五年级": "Grade 5 (Primary)",
    "六年级": "Grade 6 (Primary)",
    "初一": "Grade 7 (Junior High)",
    "初二": "Grade 8 (Junior High)",
    "初三": "Grade 9 (Junior High)",
    "高一": "Grade 10 (Senior High)",
    "高二": "Grade 11 (Senior High)",
    "高三": "Grade 12 (Senior High)",
    "小学六年级（小学毕业年，年满12岁）": "Grade 6 (Primary, Age 12)",
    "小学六年级（小学毕业年）": "Grade 6 (Primary Graduate)",
    "小学六年级（即将升入初一，年龄12岁）": "Grade 6 (Primary, transitioning to Junior High, Age 12)",
    "小学六年级毕业班（年满12岁）": "Grade 6 (Primary Graduate, Age 12)",
    "小学六年级（或等同学业水平）": "Grade 6 (Primary or equivalent)",
}

# 学科映射 - 统一标准
SUBJECT_MAP = {
    "语文": "Chinese Language",
    "数学": "Mathematics",
    "英语": "English",
    "物理": "Physics",
    "化学": "Chemistry",
    "生物": "Biology",
    "历史": "History",
    "地理": "Geography",
    "政治": "Politics",
    "道德与法治": "Morality and Law",
    "美术": "Art",
    "音乐": "Music",
    "体育": "Physical Education",
    "信息技术": "Information Technology",
    "科学": "Science",
    "书法": "Calligraphy",
    "综合实践": "Comprehensive Practice",
    "劳动与技术": "Labor and Technology",
    "思想品德": "Moral Education",
}

# 学术水平映射 - 统一标准
ACADEMIC_LEVEL_MAP = {
    "差：成绩全校排名后50%": "Poor: Bottom 50% of school",
    "低：成绩全校排名前30%至50%": "Low: Top 30%-50% of school",
    "中：成绩全校排名前10%至30%": "Average: Top 10%-30% of school",
    "高：成绩全校排名前10%": "High: Top 10% of school",
}

# 皮亚杰认知发展阶段映射 - 统一标准
PIAGET_STAGE_MAP = {
    "前运算阶段": "Preoperational Stage",
    "前运算阶段(2-7岁)": "Preoperational Stage (Ages 2-7)",
    "具体运算阶段": "Concrete Operational Stage",
    "具体运算阶段（7-11岁）": "Concrete Operational Stage (Ages 7-11)",
    "具体运算阶段（约7-11岁）": "Concrete Operational Stage (Approx. Ages 7-11)",
    "具体运算阶段（约7-12岁）": "Concrete Operational Stage (Approx. Ages 7-12)",
    "形式运算阶段": "Formal Operational Stage",
    "形式运算阶段（12岁及以上）": "Formal Operational Stage (Age 12+)",
    "形式运算阶段（约11岁及以上）": "Formal Operational Stage (Approx. Age 11+)",
    "形式运算阶段（约12岁及以上）": "Formal Operational Stage (Approx. Age 12+)",
    "形式运算阶段（初中阶段）": "Formal Operational Stage (Junior High)",
    "形式运算阶段（高中阶段）": "Formal Operational Stage (Senior High)",
}

# 埃里克森心理社会发展阶段映射 - 统一标准
ERIKSON_STAGE_MAP = {
    "勤奋对自卑": "Industry vs. Inferiority",
    "勤奋对自卑（6-12岁）": "Industry vs. Inferiority (Ages 6-12)",
    "勤奋对自卑（约6-12岁）": "Industry vs. Inferiority (Approx. Ages 6-12)",
    "勤奋对自卑阶段": "Industry vs. Inferiority Stage",
    "同一性对角色混淆": "Identity vs. Role Confusion",
    "同一性 vs 角色混淆": "Identity vs. Role Confusion",
    "同一性 vs. 角色混淆": "Identity vs. Role Confusion",
    "同一性对角色混淆阶段": "Identity vs. Role Confusion Stage",
    "同一性对角色混淆（青春期）": "Identity vs. Role Confusion (Adolescence)",
    "同一性对角色混淆（高中阶段）": "Identity vs. Role Confusion (Senior High)",
    "身份认同对角色混乱": "Identity vs. Role Confusion",
    "身份对角色混淆": "Identity vs. Role Confusion",
    "同一性 vs. 角色混乱": "Identity vs. Role Confusion",
    "自主性对羞怯怀疑": "Autonomy vs. Shame and Doubt",
}

# 科尔伯格道德发展阶段映射 - 统一标准
KOHLBERG_STAGE_MAP = {
    "前习俗水平": "Preconventional Level",
    "前习俗水平-惩罚与服从取向": "Preconventional Level - Punishment and Obedience Orientation",
    "习俗水平": "Conventional Level",
    "习俗水平（顺从权威和维护社会秩序）": "Conventional Level - Authority and Social Order Maintaining",
    "习俗水平（人际关系的协调取向）": "Conventional Level - Interpersonal Accord and Conformity",
    "常规水平": "Conventional Level",
    "后习俗水平": "Postconventional Level",
    "后习俗水平（社会契约取向）": "Postconventional Level - Social Contract Orientation",
    "后习俗水平（普遍伦理原则取向）": "Postconventional Level - Universal Ethical Principles",
}

# ==================== 翻译缓存和锁 ====================
translation_cache = {}
cache_lock = threading.Lock()

# ==================== LLM调用函数 ====================

def call_llm(text, context=""):
    """调用本地Qwen模型进行翻译"""
    if not text or len(text.strip()) == 0:
        return text
    
    # 检查缓存
    with cache_lock:
        if text in translation_cache:
            return translation_cache[text]
    
    prompt = f"""Translate the following Chinese text to English. Keep the meaning accurate and natural.
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
        translated = result["choices"][0]["message"]["content"].strip()
        
        # 保存到缓存
        with cache_lock:
            translation_cache[text] = translated
        
        return translated
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def translate_text(text, context=""):
    """翻译文本（带缓存）"""
    if not text or len(text.strip()) == 0:
        return text
    
    # 检查缓存
    with cache_lock:
        if text in translation_cache:
            return translation_cache[text]
    
    return call_llm(text, context)

# ==================== 处理函数 ====================

def translate_enum_value(value, mapping_dict, field_name):
    """使用映射表翻译枚举值，如果没有映射则使用LLM"""
    if value in mapping_dict:
        return mapping_dict[value]
    
    # 如果没有找到映射，使用LLM翻译并打印警告
    translated = translate_text(value, f"This is a {field_name} term.")
    print(f"[WARNING] No mapping for {field_name}: '{value}' -> '{translated}'")
    return translated or value

def process_record(record):
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
                
                # 根据子字段类型选择映射
                if sub_key == "皮亚杰认知发展阶段":
                    new_record[eng_key][eng_sub_key] = translate_enum_value(
                        sub_value, PIAGET_STAGE_MAP, "Piaget cognitive stage"
                    )
                elif sub_key == "埃里克森心理社会发展阶段":
                    new_record[eng_key][eng_sub_key] = translate_enum_value(
                        sub_value, ERIKSON_STAGE_MAP, "Erikson psychosocial stage"
                    )
                elif sub_key == "科尔伯格道德发展阶段":
                    new_record[eng_key][eng_sub_key] = translate_enum_value(
                        sub_value, KOHLBERG_STAGE_MAP, "Kohlberg moral stage"
                    )
                else:
                    new_record[eng_key][eng_sub_key] = sub_value
                    
        elif key == "_采样约束":
            # 处理采样约束的嵌套对象
            new_record[eng_key] = {}
            for sub_key, sub_value in value.items():
                eng_sub_key = CONSTRAINTS_MAP.get(sub_key, sub_key)
                if isinstance(sub_value, list):
                    # 翻译列表中的每个元素
                    eng_list = []
                    for item in sub_value:
                        if sub_key == "优势学科偏向":
                            eng_list.append(translate_enum_value(item, SUBJECT_MAP, "subject"))
                        else:
                            eng_list.append(item)
                    new_record[eng_key][eng_sub_key] = eng_list
                else:
                    # 翻译字符串值
                    if sub_key == "年级":
                        new_record[eng_key][eng_sub_key] = translate_enum_value(
                            sub_value, GRADE_MAP, "grade"
                        )
                    elif sub_key == "性别":
                        new_record[eng_key][eng_sub_key] = translate_enum_value(
                            sub_value, GENDER_MAP, "gender"
                        )
                    elif sub_key == "目标学术水平":
                        new_record[eng_key][eng_sub_key] = translate_enum_value(
                            sub_value, ACADEMIC_LEVEL_MAP, "academic level"
                        )
                    else:
                        new_record[eng_key][eng_sub_key] = sub_value
                    
        elif key in ["擅长科目", "薄弱科目"]:
            # 处理科目列表
            eng_list = []
            for item in value:
                eng_list.append(translate_enum_value(item, SUBJECT_MAP, "subject"))
            new_record[eng_key] = eng_list
            
        elif key in ["id"]:
            # 保持id不变
            new_record[eng_key] = value
            
        elif key == "性别":
            new_record[eng_key] = translate_enum_value(value, GENDER_MAP, "gender")
            
        elif key == "年级":
            new_record[eng_key] = translate_enum_value(value, GRADE_MAP, "grade")
            
        elif key == "学术水平":
            new_record[eng_key] = translate_enum_value(value, ACADEMIC_LEVEL_MAP, "academic level")
            
        else:
            # 其他长文本字段需要LLM翻译
            if isinstance(value, str):
                eng_value = translate_text(
                    value, 
                    f"This is a student's {eng_key} description in educational psychology context."
                )
                new_record[eng_key] = eng_value or value
            else:
                new_record[eng_key] = value
    
    return new_record

# ==================== 主函数 ====================

def main():
    input_file = "github-release-227/sample_data/merged_students_10k.jsonl"
    output_file = "github-release-227/sample_data/merged_students_10k_EN.jsonl"
    
    # 先统计行数
    print("Counting total lines...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total records to process: {total_lines}")
    print(f"Concurrency: {MAX_WORKERS}")
    
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
    
    # 并发处理
    print(f"Processing with {MAX_WORKERS} workers...")
    translated_records = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_record, record): i for i, record in enumerate(records)}
        
        with tqdm(total=len(records), desc="Translating") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    translated_record = future.result()
                    translated_records.append((futures[future], translated_record))
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
    
    print(f"\n✅ Translation complete!")
    print(f"Output file: {output_file}")
    print(f"Total cached translations: {len(translation_cache)}")

if __name__ == "__main__":
    main()
