#!/usr/bin/env python3
"""分析文件中的所有字段值，建立标准映射表"""

import json
from collections import defaultdict

def main():
    input_file = "github-release-227/sample_data/merged_students_10k.jsonl"
    
    # 收集所有唯一的值
    genders = set()
    grades = set()
    subjects = set()
    piaget_stages = set()
    erikson_stages = set()
    kohlberg_stages = set()
    academic_levels = set()
    
    print("Analyzing file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                genders.add(record.get('性别', ''))
                grades.add(record.get('年级', ''))
                academic_levels.add(record.get('学术水平', ''))
                
                # 学科
                for subj in record.get('擅长科目', []):
                    subjects.add(subj)
                for subj in record.get('薄弱科目', []):
                    subjects.add(subj)
                
                # 发展阶段
                dev_stage = record.get('发展阶段', {})
                piaget_stages.add(dev_stage.get('皮亚杰认知发展阶段', ''))
                erikson_stages.add(dev_stage.get('埃里克森心理社会发展阶段', ''))
                kohlberg_stages.add(dev_stage.get('科尔伯格道德发展阶段', ''))
                
            except Exception as e:
                continue
    
    print("\n=== Gender ===")
    for v in sorted(genders):
        print(f"  '{v}': '',")
    
    print("\n=== Grade ===")
    for v in sorted(grades):
        print(f"  '{v}': '',")
    
    print("\n=== Subjects ===")
    for v in sorted(subjects):
        print(f"  '{v}': '',")
    
    print("\n=== Piaget Stages ===")
    for v in sorted(piaget_stages):
        print(f"  '{v}': '',")
    
    print("\n=== Erikson Stages ===")
    for v in sorted(erikson_stages):
        print(f"  '{v}': '',")
    
    print("\n=== Kohlberg Stages ===")
    for v in sorted(kohlberg_stages):
        print(f"  '{v}': '',")
    
    print("\n=== Academic Levels ===")
    for v in sorted(academic_levels):
        print(f"  '{v}': '',")

if __name__ == "__main__":
    main()
