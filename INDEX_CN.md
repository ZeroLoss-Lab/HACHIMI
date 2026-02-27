# HACHIMI 代码仓库索引

<p align="center">
  <a href="INDEX.md">English</a> | <a href="INDEX_CN.md">简体中文</a>
</p>



## 📁 文件结构

```
github-release/
├── README.md                   # 主要文档（英文）
├── README_CN.md                # 主要文档（中文）
├── INDEX.md                    # 本文件（英文）
├── INDEX_CN.md                 # 本文件（中文）
├── LICENSE                     # MIT 许可证
├── .gitignore                  # Git 忽略模式
│
├── requirements.txt            # Python 依赖
├── rules.py                    # 15 条验证规则 (R1-R15)
├── providers.py                # API 提供商管理与速率限制
│
├── app.py                      # 【主程序】Streamlit Web UI (Qwen2.5-72B)
├── app_cli.py                  # 命令行界面
├── app_for_GPT4.1.py           # GPT-4.1 优化版本
│
├── baseline_single_shot.py     # 单次基线用于对比
├── analyse_all.py              # 分析工具（完整数据集）
├── analyse_by_chunk.py         # 分析工具（分块数据）
├── provider_health_check.py    # API 提供商健康检查
├── test_field_mapping.py       # 字段映射单元测试
│
├── sample_data/
│   └── merged_students_10k.jsonl   # 10,000 条匿名化学生画像
│
└── secrets/
    └── README.md               # API 配置指南
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

创建 `secrets/providers.json`（格式见 `secrets/README.md`）：

```json
{
  "providers": [
    {
      "name": "your_provider",
      "base_url": "https://api.example.com/v1",
      "api_key": "your-api-key",
      "model": "model-name",
      "qpm": 60
    }
  ]
}
```

### 3. 运行系统

**Web UI（推荐）：**
```bash
streamlit run app.py
```

**命令行：**
```bash
python app_cli.py --count 100 --grade 初一 --gender 女
```

---

## 📋 文件说明

| 文件 | 用途 | 运行方式 |
|------|---------|------------|
| `app.py` | 主程序 - 多智能体协作生成（Web UI） | `streamlit run app.py` |
| `app_cli.py` | CLI 版本 - 批量生成 | `python app_cli.py` |
| `app_for_GPT4.1.py` | GPT-4.1 优化版本 | `streamlit run app_for_GPT4.1.py` |
| `baseline_single_shot.py` | 基线 - 单次调用生成 | `python baseline_single_shot.py` |
| `providers.py` | 提供商池管理（被其他文件导入） | 不可直接运行 |
| `rules.py` | 15 条验证规则（被其他文件导入） | 不可直接运行 |
| `analyse_all.py` | 分析生成结果（完整） | `python analyse_all.py` |
| `analyse_by_chunk.py` | 分析生成结果（分块） | `python analyse_by_chunk.py` |
| `provider_health_check.py` | 检查 API 提供商可用性 | `python provider_health_check.py` |
| `test_field_mapping.py` | 测试字段映射正确性 | `python test_field_mapping.py` |

---

## 🔧 核心特性

### 多智能体架构
- **5 个内容智能体**: 入学与发展、学业画像、人格与价值观、社交与创造力、心理健康
- **2 个验证器**: 快速验证器 + 深度验证器
- **多轮协商**: 可配置的细化轮数

### 质量控制
- **15 条验证规则 (R1-R15)**: 年龄-年级一致性、发展阶段对齐、跨维度一致性等
- **SimHash 去重**: 海明距离阈值以确保多样性
- **学业水平锚定**: 严格四选一格式（高/中/低/差）

### 采样控制
- **QuotaScheduler**: 年级 × 性别 × 科目簇预采样
- **轻量过滤**: 实时质量筛查
- **乐观偏置抑制**: 跨维度一致性检查

---

## 📊 样例数据

`sample_data/merged_students_10k.jsonl` 包含 10,000 条匿名化学生画像，字段包括：
- 基本信息：id、年龄、性别、年级、agent_name
- 发展阶段：皮亚杰认知 / 埃里克森心理社会 / 科尔伯格道德
- 学业信息：优势/劣势科目、学业水平
- 心理画像：人格、价值观、社交关系、创造力、心理健康

---

## ⚠️ 重要说明

1. **API 配置**: 运行前必须配置 `secrets/providers.json`
2. **速率限制**: 内置令牌桶速率限制 - 根据你的 API 配额调整 `qpm`
3. **分块存储**: 结果自动以 50 条记录为一块保存到 `output/<run_id>/`
4. **匿名化**: 所有生成数据自动将姓名替换为 "X号学生" 格式

---

## 📄 引用

```bibtex
@inproceedings{jiang2026hachimi,
  title={Generating Authentic Student Profiles: A Multi-Agent Collaboration Approach},
  author={Jiang, Yilin and Tan, Fei and Yin, Xuanyu and Leng, Jing and Zhou, Aimin},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026}
}
```

---

**作者**: 姜逸林、谭斐（通讯作者）、尹轩宇、冷晶、周爱民  
**单位**: 华东师范大学、港科大(广州)、上海创智学院  
**联系**: ftan@mail.ecnu.edu.cn

---

> 🔔 **开源声明**: 完整数据集和源代码即将发布，敬请期待！
