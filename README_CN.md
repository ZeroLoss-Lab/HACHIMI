# HACHIMI: 基于编排智能体的可扩展且可控的学生画像生成

<p align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">简体中文</a>
</p>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HACHIMI 是一个多智能体协作系统，利用大型语言模型（LLM）生成全面的学生画像。该系统通过复杂的五智能体协作框架，创建详细的学生人设，包括学业表现、人格特质、价值观、创造力和心理健康特征。

## 🌟 核心特性

- **多智能体协作**: 五个基于 LLM 的专业智能体在共享白板上协同工作
- **全面画像**: 生成 15 个以上维度的、具有高度一致性的学生画像
- **验证器架构**: 两阶段验证（快速验证器 + 深度验证器），支持可配置的协商轮数
- **智能去重**: 基于 SimHash 的相似度过滤以确保多样性
- **自适应速率限制**: 采用 AIMD（加法增大、乘法减小）算法进行 API 配额管理
- **多提供商支持**: 跨多个 LLM API 提供商的自动负载均衡
- **生产就绪**: CLI 和 Streamlit UI 模式，支持自动检查点保存和增量生成

## 📋 目录

1. [架构概览](#-架构概览)
2. [安装](#-安装)
3. [快速开始](#-快速开始)
4. [配置](#-配置)
5. [使用](#-使用)
6. [输出格式](#-输出格式)
7. [评估](#-评估)
8. [基线系统](#-基线系统)
9. [文档](#-文档)
10. [引用](#-引用)
11. [许可证](#-许可证)

## 🏗️ 架构概览

### 多智能体系统

HACHIMI 使用五个在共享白板上协作的专业智能体：

1. **入学与发展智能体**: 处理基本信息、发展阶段
2. **学业画像智能体**: 生成学业表现、优势/劣势
3. **人格与价值观智能体**: 创建人格特质和价值体系
4. **社交与创造力智能体**: 构建社交关系和创造力画像
5. **心理健康智能体**: 生成心理健康评估

### 系统组件

- **`app.py`**: Qwen2.5-72B 版本，带 Streamlit 前端
- **`app_for_GPT4.1.py`**: GPT-4.1 版本，带 Streamlit 前端
- **`app_cli.py`**: 用于无头生成的命令行界面
- **`providers.py`**: 提供商池管理，带 AIMD 速率限制
- **`rules.py`**: 15 条全面的验证规则（年龄-年级对齐、一致性检查等）
- **`baseline_single_shot.py`**: 用于对比的单次基线系统
- **`analyse_all.py`** / **`analyse_by_chunk.py`**: 质量评估工具

## 📥 安装

### 要求

- Python 3.8 或更高版本
- 10GB+ 可用磁盘空间（用于生成的画像）
- 访问 LLM API 端点（OpenAI 兼容）

### 安装步骤

```bash
# 下载匿名提交的仓库
# 点击 Anonymous GitHub 上的 "Download repository" 按钮

# 安装依赖
pip install -r requirements.txt

# 配置 API 提供商
nano secrets/providers.json  # 添加你的 API 端点
```

### 依赖

```
streamlit>=1.28.0
requests>=2.28.0
```

标准库模块：`json` `re` `random` `math` `os` `glob` `time` `hashlib` `threading` `concurrent.futures`

## 🚀 快速开始

### 方式 1: CLI（推荐用于大规模生成）

```bash
# 默认生成（100 万条记录，可中断并恢复）
python app_cli.py

# 配置通过编辑 app_cli.py 顶部的变量完成：
# - RUN_ID: "my_experiment"
# - TOTAL: 1000000
# - CHUNK_SIZE: 50
# - MAX_ROUNDS: 3
# - SIMHASH: 3（海明距离阈值）
# - KEYS_PATH: "secrets/providers.json"
```

### 方式 2: 使用 Qwen2.5-72B 的 Streamlit UI（交互式）

```bash
streamlit run app.py
```

然后在侧边栏配置：
- 要生成的记录总数
- 学业水平分布
- SimHash 阈值
- 最大协商轮数

### 方式 3: 使用 GPT-4.1 的 Streamlit UI（交互式）

```bash
streamlit run app_for_GPT4.1.py
```

通过侧边栏配置（与方式 2 相同）。此版本针对 GPT-4.1 API 进行了优化。

### 首次设置

1. **配置 API 提供商**

编辑 `secrets/providers.json`：
```json
[
  {
    "name": "Provider-Name",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-your-key-here",
    "model": "gpt-4",
    "qpm": 100,
    "capacity_max": 150
  }
]
```

2. **测试提供商连接**

```bash
python -c "from providers import load_providers; clients = load_providers('secrets/providers.json'); print(f'Loaded {len(clients)} providers')"
```

## ⚙️ 配置

### API 提供商配置

HACHIMI 支持多种提供商格式：

**JSON 格式：**
```json
{"name":"Provider","base_url":"https://api.example.com/v1","api_key":"sk-key","model":"gpt-4","qpm":100}
```

**管道分隔格式：**
```
Provider|https://api.example.com/v1|sk-key|gpt-4|qpm=100|capacity_max=150
```

**关键参数：**
- `name`: 提供商标识符
- `base_url`: API 端点（OpenAI 兼容）
- `api_key`: 认证密钥
- `model`: 模型名称
- `qpm`: 每分钟查询数（影响速率限制）
- `capacity_max`: 自适应扩展的最大容量

### 生成参数

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `TOTAL` | 1,000,000 | 要生成的记录总数 |
| `CHUNK_SIZE` | 50 | 每个输出文件的记录数 |
| `MAX_ROUNDS` | 3 | 验证器-智能体协商的最大轮数 |
| `SIMHASH` | 3 | 去重的海明距离阈值 |
| `PER_STEP_M` | 20 | 调度的批次大小（不影响速度） |

### 关于样例数据

本次发布包含**一个数据集**：

- **10K 样例** (`sample_data/merged_students_10k.jsonl`, 49MB): 我们系统生成的真实画像，用于**审稿人检查**和**快速测试**。

此样例用于在同行评审期间促进可重复性检查。**完整数据集（100 万+ 画像）**将在**评审流程后**通过**我们的平台**发布。

### 性能调优

生成速度主要由以下因素控制：
1. **提供商数量**: `secrets/providers.json` 中的提供商数量
2. **每个提供商的 QPM**: 每个提供商配置的每分钟查询数
3. **API 延迟**: API 端点的实际响应时间
4. **MAX_WORKERS**（可选）: 手动并发限制

**不受以下因素影响：**
- `PER_STEP_M`: 仅影响批次大小，不影响吞吐量
- `TOTAL`: 仅决定总运行时间，不影响速率
- `CHUNK_SIZE`: 仅影响输出文件组织

有关详细的速度优化指南，请参阅本 README 中的提供商配置和性能调优部分。

## 📊 使用

### 基本命令

选择你喜欢的界面：

**命令行（无头模式）**
```bash
# 不使用 UI 生成画像
python app_cli.py

# 实时监控进度
tail -f output/*/students_chunk_*.jsonl
```

**交互式 UI（Qwen2.5-72B 版本）**
```bash
streamlit run app.py
```

**交互式 UI（GPT-4.1 版本）**
```bash
streamlit run app_for_GPT4.1.py
```

**评估**
```bash
# 评估所有运行
python analyse_by_chunk.py

# 评估特定运行
python analyse_by_chunk.py --input_dir ./output/run_experiment_001

# 评估合并输出
python analyse_all.py --simhash_threshold 3 --topk 100

# 运行基线对比
python baseline_single_shot.py --mode age_15 --count 5000 --output ./output/baseline
```

### 分析和评估

```bash
# 对所有输出运行质量分析
python analyse_by_chunk.py

# 对比两次运行
python compare_baseline.py --group1 ./output/multi_agent --group2 ./output/baseline

# 评估样例数据（10K 画像）
python analyse_all.py --input_file ./sample_data/merged_students_10k.jsonl --simhash_threshold 3
```

## 📤 输出格式

### 目录结构

```
./output/
  {run_id}/
    students_chunk_1.jsonl      # 每块 50 条记录
    students_chunk_2.jsonl
    ...
    schedule.json               # 带采样约束的生成计划
    meta.json                   # Total_N 和 chunk_size
    failures.jsonl              # 失败的生成尝试

./eval/
  overview.md                   # 所有运行的摘要
  _index.csv                    # 索引表
  _index.json                   # 索引数据
  {run_id}/
    summary.json                # 指标和分布
    per_item.jsonl              # 每条记录的验证结果
    near_duplicates.json        # 相似画像对
    suspicious_samples.json     # 问题最多的样例
    dashboard.md                # 人类可读报告
    buckets.csv                 # 分布明细
```

### 学生画像模式

```json
{
  "id": 1,
  "name": "学生姓名",
  "age": 15,
  "gender": "女",
  "grade": "初三",
  "agent_name": "zhang1_mei3",
  "developmental_stages": {
    "piaget": "形式运算阶段",
    "erikson": "身份与角色混淆",
    "kohlberg": "后习俗水平"
  },
  "strong_subjects": ["数学", "物理"],
  "weak_subjects": ["历史", "政治"],
  "academic_level": "高：成绩全校排名前10%",
  "personality": "认真负责且内向...",
  "values": "展现出强烈的道德品质...",
  "social_relationships": "保持亲密的友谊...",
  "creativity": "在问题解决中表现出适度的流畅性...",
  "mental_health": "总体适应良好，压力水平正常..."
}
```

### 学业水平（严格四选一）

- "高：成绩全校排名前10%"
- "中：成绩全校排名前10%至30%"
- "低：成绩全校排名前30%至50%"
- "差：成绩全校排名后50%"

## 📈 评估

### 质量指标

- **Distinct-1/2**: 字符级文本多样性
- **SimHash 海明距离**: 近似重复检测
- **模板检测**: 跨维度的 Jaccard 相似度

### 验证覆盖

15 条全面规则涵盖：
- ✅ 必填字段存在性
- ✅ 年龄-年级对齐（规则 R1）
- ✅ 发展阶段一致性（规则 R2）
- ✅ 学业水平格式
- ✅ 科目集合不重叠（规则 R3）
- ✅ 创造力可行性一致性（规则 R4）
- ✅ 价值观-心理健康对齐（规则 R5）
- ✅ 代理名格式（规则 R6）
- ✅ 7 维度价值观覆盖（规则 R8）
- ✅ 8 维度创造力覆盖（规则 R9）
- ✅ 心理健康结构（规则 R10）
- ✅ 跨字段一致性（规则 R11）
- ✅ 非诊断化语言（规则 R12）
- ✅ 自然语言（规则 R13）
- ✅ 单段格式（规则 R14）

### 样例评估结果

10K 画像的预期质量指标：
- 通过率：~95-98%
- Distinct-1: 0.15-0.25
- Distinct-2: 0.40-0.60
- 生成速度：每秒 2-5 个画像（取决于 API）

## 📚 文档

- **[CLAUDE.md](CLAUDE.md)**: 开发指南和架构概览
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)**: 安装指南
- **[Validation Rules](rules.py)**: 完整的验证规则规范

## 🔬 引用

如果你在研究中使用本系统，请引用：

```bibtex
@software{hachimi2025,
  title={HACHIMI: Human-centric Agent-based Character and Holistic Individual Modeling Infrastructure},
  author={Research Team},
  year={2025},
  note={Multi-agent LLM system for comprehensive student profile generation}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 基于 OpenAI 兼容的 LLM API 构建
- 使用多智能体 AI 架构和 SimHash 进行高效重复检测
- 基于 AIMD（加法增大、乘法减小）的速率限制算法
