#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试字段映射功能：英文字段 -> 中文字段
"""
import json
import sys
import os

# 添加项目路径
sys.path.append('d:/Study/sii/StudentGenerate/StudentGenerate/HACHIMI-Human-centric-Agent-based-Character-and-Holistic-Individual-Modeling-Infrastructure')

# 直接导入相关变量/函数
from app import try_json, FIELD_NAME_MAPPING

print('=' * 80)
print('字段映射测试：英文字段名 -> 中文字段名')
print('=' * 80)

print('\n字段映射表片段：')
for i, (en, cn) in enumerate(list(FIELD_NAME_MAPPING.items())[:10]):
    print(f'  {en} -> {cn}')

print('\n' + '=' * 80)
print('测试用例')
print('=' * 80)

# 测试用例
test_cases = [
    {
        "name": "测试用例 1 - 简单学生信息",
        "data": '''{
  "id": 1,
  "name": "张三",
  "age": 15,
  "grade": "高二",
  "gender": "男"
}''',
        "expected": ["id", "姓名", "年龄", "年级", "性别"]
    },
    {
        "name": "测试用例 2 - 完整学生信息（多种英文字段）",
        "data": '''{
  "id": 2,
  "name": "李四",
  "age": 13,
  "grade": "初一",
  "gender": "女",
  "development_stage": {"皮亚杰认知发展阶段": "形式运算阶段"},
  "advantageSubjects": ["数学", "物理"],
  "weakSubjects": ["英语", "生物"],
  "academicLevel": "中：成绩全校排名前10%至30%"
}''',
        "expected": ["id", "姓名", "年龄", "年级", "性别", "发展阶段", "擅长科目", "薄弱科目", "学术水平"]
    },
    {
        "name": "测试用例 3 - 别名字段",
        "data": '''{
  "id": 3,
  "student_id": 3,
  "persona": "开朗",
  "morals": "道德价值观",
  "psychological": "心理健康概述",
  "socialRelationships": "社交关系描述"
}''',
        "expected": ["id", "id", "人格", "价值观", "心理健康", "社交关系"]
    },
    {
        "name": "测试用例 4 - 包含说明文字的 JSON 提取",
        "data": '''这是一些说明文字：
{
  "name": "王五",
  "age": 16,
  "gender": "男"
}
更多说明文字...''',
        "expected": ["姓名", "年龄", "性别"]
    },
    {
        "name": "测试用例 5 - Agent 返回的完整结构",
        "data": '''{
  "name": "赵六",
  "agent_name": "zhao4_liu4",
  "age": 14,
  "grade": "初二",
  "gender": "女",
  "development": {
    "皮亚杰认知发展阶段": "形式运算阶段",
    "埃里克森心理社会发展阶段": "同一性对角色混乱",
    "科尔伯格道德发展阶段": "习俗水平"
  }
}''',
        "expected": ["姓名", "代理名", "年龄", "年级", "性别", "发展阶段"]
    },
    {
        "name": "测试用例 6 - 无效内容",
        "data": "这不是一个有效的 JSON 字符串",
        "expected": []
    }
]

# 运行测试
passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    print(f'\n{i}. {test["name"]}')
    print('-' * 80)

    # 执行转换
    result = try_json(test["data"])

    # 显示原始输入（前 100 字符）
    print(f'原始: {test["data"][:100]}...')
    print(f'结果字段: {list(result.keys())}')
    print(f'预期字段: {test["expected"]}')

    # 验证
    result_keys = set(result.keys())
    expected_keys = set(test["expected"])

    # 对于空预期，检查是否返回空字典
    if len(test["expected"]) == 0:
        if len(result) == 0:
            print('✅ 通过：正确返回空结果')
            passed += 1
        else:
            print(f'❌ 失败：预期空结果，实际得到 {list(result.keys())}')
            failed += 1
    else:
        # 检查是否所有预期字段都存在
        if expected_keys.issubset(result_keys):
            print('✅ 通过：所有预期字段都已转换')
            passed += 1
        else:
            missing = expected_keys - result_keys
            print(f'❌ 失败：缺少字段 {missing}')
            failed += 1

    # 显示转换后的完整结果
    if result:
        print(f'转换结果:\n{json.dumps(result, ensure_ascii=False, indent=2)}')

print('\n' + '=' * 80)
print(f'测试完成: 通过 {passed} / 失败 {failed} / 总计 {passed + failed}')
print('=' * 80)

if failed > 0:
    sys.exit(1)