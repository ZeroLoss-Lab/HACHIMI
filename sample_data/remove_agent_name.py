#!/usr/bin/env python3
"""删除原始文件中的代理名字段"""

import json

input_file = "merged_students_10k.jsonl"
output_file = "merged_students_10k_no_agent.jsonl"

count = 0
with open(input_file, 'r', encoding='utf-8', errors='ignore') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        try:
            record = json.loads(line.strip())
            # 删除代理名字段
            if "代理名" in record:
                del record["代理名"]
                count += 1
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error: {e}")
            continue

print(f"Removed '代理名' from {count} records")
print(f"Output: {output_file}")
