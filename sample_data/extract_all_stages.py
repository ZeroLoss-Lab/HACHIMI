#!/usr/bin/env python3
"""提取所有发展阶段和学术水平的唯一值"""

import json
from collections import defaultdict

def main():
    input_file = "github-release-227/sample_data/merged_students_10k.jsonl"
    
    # 收集所有唯一的值
    piaget_stages = defaultdict(int)
    erikson_stages = defaultdict(int)
    kohlberg_stages = defaultdict(int)
    academic_levels = defaultdict(int)
    grades = defaultdict(int)
    
    print("Analyzing file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                
                # 发展阶段
                dev_stage = record.get('发展阶段', {})
                if dev_stage.get('皮亚杰认知发展阶段'):
                    piaget_stages[dev_stage['皮亚杰认知发展阶段']] += 1
                if dev_stage.get('埃里克森心理社会发展阶段'):
                    erikson_stages[dev_stage['埃里克森心理社会发展阶段']] += 1
                if dev_stage.get('科尔伯格道德发展阶段'):
                    kohlberg_stages[dev_stage['科尔伯格道德发展阶段']] += 1
                
                # 学术水平
                if record.get('学术水平'):
                    academic_levels[record['学术水平']] += 1
                    
                # 年级
                if record.get('年级'):
                    grades[record['年级']] += 1
                    
            except Exception as e:
                continue
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} records...")
    
    print(f"\n=== Piaget Stages ({len(piaget_stages)} unique) ===")
    for v, c in sorted(piaget_stages.items(), key=lambda x: -x[1]):
        print(f"  '{v}': '',  # {c}")
    
    print(f"\n=== Erikson Stages ({len(erikson_stages)} unique) ===")
    for v, c in sorted(erikson_stages.items(), key=lambda x: -x[1]):
        print(f"  '{v}': '',  # {c}")
    
    print(f"\n=== Kohlberg Stages ({len(kohlberg_stages)} unique) ===")
    for v, c in sorted(kohlberg_stages.items(), key=lambda x: -x[1]):
        print(f"  '{v}': '',  # {c}")
    
    print(f"\n=== Academic Levels ({len(academic_levels)} unique) ===")
    for v, c in sorted(academic_levels.items(), key=lambda x: -x[1]):
        print(f"  '{v}': '',  # {c}")
    
    print(f"\n=== Grades ({len(grades)} unique) ===")
    for v, c in sorted(grades.items(), key=lambda x: -x[1]):
        print(f"  '{v}': '',  # {c}")

if __name__ == "__main__":
    main()
