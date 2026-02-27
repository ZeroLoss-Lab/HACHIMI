# -*- coding: utf-8 -*-
"""
baseline_single_shot.py - 单一大模型一次性生成学生画像基线版本
作为多智能体协作系统的对照组，直接请求大模型生成完整的画像

使用方法:
    # 生成5000条年龄=15的学生
    python baseline_single_shot.py --mode age_15 --count 5000 --output ./output/baseline_age15

    # 生成5000条年级=初二的学生
    python baseline_single_shot.py --mode grade_chu2 --count 5000 --output ./output/baseline_chu2

特性:
- 单一prompt覆盖所有字段生成
- 使用Qwen2.5-72B模型
- 每次生成传入随机seed确保多样性
- 支持年龄筛选和年级筛选
"""

import json
import os
import sys
import time
import argparse
import random
import re
from typing import Dict, Any, List, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import Counter
import glob

# 导入provider配置和限流逻辑
from providers import load_providers, ProviderPool, ProviderClient, set_client_for_thread

# ================== 配置常量 ==================
LEVELS = [
    "高：成绩全校排名前10%",
    "中：成绩全校排名前10%至30%",
    "低：成绩全校排名前30%至50%",
    "差：成绩全校排名后50%",
]

GRADE_AGE_MAP = {
    "一年级": (6, 7),
    "二年级": (7, 8),
    "三年级": (8, 9),
    "四年级": (9, 10),
    "五年级": (10, 11),
    "六年级": (11, 12),
    "初一": (12, 13),
    "初二": (13, 14),
    "初三": (14, 15),
    "高一": (15, 16),
    "高二": (16, 17),
    "高三": (17, 18),
}

GRADES = ["一年级","二年级","三年级","四年级","五年级","六年级","初一","初二","初三","高一","高二","高三"]
GENDERS = ["男","女"]

# 字段验证规则
AGENT_ID_REGEX = r"^(?:[a-z]+[1-5]){1,2}_(?:[a-z]+[1-5]){1,3}$"

# ================== 核心生成逻辑 ==================
class SingleShotGenerator:
    def __init__(self, provider_pool: ProviderPool):
        self.pool = provider_pool
        self.client_map = {}
        self.lock = threading.Lock()

    def _get_thread_client(self) -> ProviderClient:
        """每个线程获取自己的client"""
        tid = threading.get_ident()
        with self.lock:
            if tid not in self.client_map:
                self.client_map[tid] = self.pool.pick()
            return self.client_map[tid]

    def _generate_seed(self) -> str:
        """生成随机种子用于prompt"""
        return f"SEED-{random.randrange(10**16, 10**17-1)}"

    def _build_single_prompt(self, constraint_info: str, seed: str, target_level: Optional[str] = None) -> str:
        """构建一次性生成完整画像的prompt"""

        # 学术水平分布提示
        level_dist_hint = ""
        if target_level:
            level_dist_hint = f"\n【重要约束】学术水平必须严格设置为：{target_level}"
        else:
            level_dist_hint = (
                "\n为保证样本多样性，请按以下比例分布学术水平："
                "\n  - 高：25%"
                "\n  - 中：25%"
                "\n  - 低：25%"
                "\n  - 差：25%"
            )

        prompt = f"""你是一个学生画像生成器。请一次性生成一个完整、真实的中国学生画像。

**核心要求：**
1. **一锤子输出**：所有字段必须在一次响应中完整生成，不得分步或迭代
2. **格式严格**：输出**仅包含一个JSON对象**，无任何额外文字说明，不使用```json或```包裹
3. **所有字段必填**：必须有id、姓名、年龄、性别、年级、学术水平、人格、社交关系、擅长科目、薄弱科目、发展阶段、代理名、价值观、创造力、心理健康
4. **自然真实**：既有优点也有不足，避免全是"优秀""非常好"等极端正面评价

**画像字段与约束：**

{constraint_info}
{level_dist_hint}

**姓名**：中文名，2-4个字，符合年龄段特点

**年龄**：整数，中国学生典型年龄范围

**性别**：只能是"男"或"女"

**年级**：从"一年级"到"高三"之一，必须与年龄匹配（允许±1岁偏差）

**代理名**：必须符合正则 {AGENT_ID_REGEX}
- 格式：姓1-2音节_名1-3音节
- 每个音节：小写拼音+声调数字(1-5)
- 示例：zhang1_shuang3, li1_huan4ying1, ou3yang2_ming2hao3

**学术水平**：严格四选一（以下四个字符串之一，不能自己编造）：
- "高：成绩全校排名前10%"
- "中：成绩全校排名前10%至30%"
- "低：成绩全校排名前30%至50%"
- "差：成绩全校排名后50%"

**擅长科目与薄弱科目**：
- 两个非空数组，元素为学科名称（如"语文"、"数学"）
- 集合不相交（同一科目不能在两个数组中）

**发展阶段**：对象，包含三个字段：
- "皮亚杰认知发展阶段"：与年龄匹配（12岁以下是"具体运算阶段"，12岁以上是"形式运算阶段"）
- "埃里克森心理社会发展阶段"：与年龄匹配（6-12岁是"勤奋对自卑阶段"，12-18岁是"身份对角色混乱阶段"）
- "科尔伯格道德发展阶段"：与年龄匹配（10岁以下是"前习俗水平"，10-15岁是"习俗水平"，15岁以上可能是"后习俗水平"）

**人格**：1-2句自然语言，概括核心人格特质（如内向/外向、责任感、开放性等）

**价值观**：**单段连续自然语言**（不得分段或使用列表符号）
- 必须显式覆盖七个维度：道德修养、身心健康、法治意识、社会责任、政治认同、文化素养、家庭观念
- 每个维度都要有可识别的等级词（高/较高/中上/中/较低/低）
- **禁止**全部维度都是"较高/很高"，必须有明显的优劣势分布（中/较低/低至少出现2次）

**社交关系**：**单段自然语言**（约160-260字）
- 按"背景→关键事件→影响"顺序叙述
- 不得换行或条列
- 体现真实的学生社交特点

**创造力**：**单段连续自然语言**（不得分段或使用列表符号）
- 必须包含八维度的评价：流畅性、新颖性、灵活性、可行性、问题发现、问题分析、提出方案、改善方案
- 每个维度都有清晰的等级词(高/较高/中上/中/较低/低)和简短依据
- 结尾有一个整体"雷达总结"概括（必须有"雷达"或"总结"词）
- **禁止**8个维度全部是"高/较高"，必须有劣势项（中/较低/低至少出现2次）
- 内部一致性：若"可行性"较低/低，则"提出方案"不能高于"中"

**心理健康**：**单段连续自然语言**（不得分段或使用列表符号）
- 按照以下顺序在段内自然穿插：
  1) 概述整体心理状态
  2) 至少两点性格特征，与心理适应相关
  3) 对以下项目给出明确等级或程度：综合心理状况、幸福指数、抑郁风险、焦虑风险
  4) 如无明确心理疾病，需包含"信息不足或未见显著症状"等非诊断化表述
  5) 简短背景故事（学习压力、人际冲突、家庭事件等）
  6) 目前的支撑与应对方式（家庭、老师、同伴、学校资源）
- **禁止**使用"重度抑郁/双相/用药/住院"等重临床表述，允许"倾向/轻度/阶段性/可管理/建议咨询"

**多样性种子**：{seed}
- 请充分利用该种子生成独特、多样的学生画像，避免模板化和重复

**输出格式示例（请严格遵循此结构）：**
{{
  "id": 1,
  "姓名": "张爽",
  "年龄": 15,
  "性别": "男",
  "年级": "初三",
  "代理名": "zhang1_shuang3",
  "学术水平": "中：成绩全校排名前10%至30%",
  "擅长科目": ["语文", "历史"],
  "薄弱科目": ["数学", "物理"],
  "发展阶段": {{
    "皮亚杰认知发展阶段": "形式运算阶段",
    "埃里克森心理社会发展阶段": "身份对角色混乱阶段",
    "科尔伯格道德发展阶段": "习俗水平"
  }},
  "人格": "性格温和细腻，有较强的责任感和同理心，做事稳重但在陌生情境中略显拘谨。",
  "价值观": "在道德修养上整体较高，能够主动遵守班级规则并在同伴发生冲突时尝试调解，但在面对复杂情境时偶尔会犹豫不决；身心健康处于中等水平，平时能通过运动和写日记调节情绪，但考试前仍会感到明显紧张；法治意识较为清晰，能够理解简单的法律和校规边界，对网络诈骗等有一定警惕；社会责任感中等，愿意在班级和社区活动中承担力所能及的任务但主动性需进一步提升；在政治认同方面能理解并认同国家和集体基本价值；文化素养中等，阅读面尚可但对历史与文学作品的理解深度有限；家庭观念较为稳定，以父母为主要支持来源，能在尊重父母意见的同时尝试表达自我，但在重大决策上仍较为依赖家庭。",
  "社交关系": "在班级中整体人缘较好，喜欢与同桌和小组同学分享学习心得，但在陌生同学面前起初略显拘谨。一次小组项目中，他在分工时主动协调不同意见，帮助同伴达成共识，这件事让他意识到自己在倾听与沟通上的优势，也增强了与组员之间的信任感。",
  "创造力": "在流畅性上中上水平，能够较快提出多种备选想法；新颖性中等，偶尔能跳出现有范式提出有亮点的点子；灵活性中等偏下，在遇到完全陌生情境时需要较长时间才能转换思路；可行性整体较高，倾向于把想法落实为可执行的步骤；在问题发现与问题分析上表现较好，能够抓住任务中的关键矛盾并用简洁语言解释；提出方案的能力与可行性匹配，多数建议务实具体；改善方案方面略显保守，对既有方案的迭代更多停留在细节层面。总体雷达来看，他的创造力在"发现与落实"链条上较为突出，而在"突破与变通"上仍有成长空间。",
  "心理健康": "整体综合心理状况处于中等偏上水平，日常情绪大多稳定，但在成绩波动或重要考试阶段会出现短时紧张和自我怀疑。性格上既有一定的敏感细腻，也愿意向信任的同伴和家人倾诉。主观幸福指数中等，能从阅读、音乐和运动中获得放松。当前抑郁风险和焦虑风险整体偏低，偶尔出现的情绪低落多与学习负担和对自我要求较高有关，信息不足或未见显著症状指向临床意义上的抑郁或焦虑障碍。家庭氛围较为支持，父母在学习期待上略高但会尝试调整沟通方式；学校中与班主任和几位科任老师关系良好，愿意在需要时寻求帮助。整体来看，只要保持现有支持系统，并逐步学习更灵活的时间管理和情绪调节方式，他在身心健康上的发展空间仍然较大。"
}}

**重要提醒：**
1. **只输出JSON对象**，不要任何额外文字或Markdown符号
2. **所有字段必须存在且非空**
3. **必须包含多样性种子，避免生成相似内容**
4. **年龄-年级-发展阶段必须一致**
5. **学术水平严格四选一**
6. **价值观/创造力/心理健康必须是单段文本，不得出现\\n\\n或列表符号**
7. **代理名必须符合正则规则**
"""
        return prompt

    def _call_llm(self, prompt: str, max_tokens: int = 4096, temperature: float = 1.0) -> str:
        """调用LLM接口"""
        client = self._get_thread_client()
        set_client_for_thread(client)

        messages = [
            {"role": "system", "content": "你是一个学生画像生成器，严格按照要求输出JSON格式的学生画像，不需要任何解释文字。"},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            data = client.post("/chat/completions", payload, timeout=180)
            # 尝试多种方式解析content
            content = None
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")

                # 处理list类型的content（某些模型会返回[{"type":"text","text":"..."}]）
                if isinstance(content, list):
                    segs = []
                    for part in content:
                        if isinstance(part, dict):
                            if "text" in part:
                                segs.append(part["text"])
                            elif "content" in part:
                                segs.append(part["content"])
                        elif isinstance(part, str):
                            segs.append(part)
                    content = "".join(segs)

            if not content:
                content = json.dumps(data, ensure_ascii=False)

            return content

        except Exception as e:
            print(f"[ERROR] LLM调用失败: {e}")
            raise

    def _parse_json_output(self, text: str) -> Optional[Dict[str, Any]]:
        """解析LLM输出的JSON"""
        if not text:
            return None

        text = text.strip()

        # 去掉```json和```包裹
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 尝试直接解析
        try:
            obj = json.loads(text)
            return obj
        except:
            pass

        # 尝试提取第一个{...}
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj
            except:
                pass

        return None

    def _validate_structure(self, item: Dict[str, Any]) -> tuple[bool, List[str]]:
        """轻量级结构验证"""
        errors = []

        # 必填字段
        required = ["id", "姓名", "年龄", "性别", "年级", "学术水平", "人格",
                   "社交关系", "擅长科目", "薄弱科目", "发展阶段", "代理名",
                   "价值观", "创造力", "心理健康"]

        for field in required:
            if field not in item or not item[field]:
                errors.append(f"缺失字段: {field}")

        # 学术水平验证
        if "学术水平" in item and item["学术水平"] not in LEVELS:
            errors.append(f"学术水平非四选一: {item.get('学术水平')}")

        # 代理名验证
        if "代理名" in item:
            if not re.match(AGENT_ID_REGEX, str(item["代理名"])):
                errors.append(f"代理名格式错误: {item.get('代理名')}")

        # 性别验证
        if "性别" in item and item["性别"] not in ["男", "女"]:
            errors.append(f"性别错误: {item.get('性别')}")

        # 年级验证
        if "年级" in item and item["年级"] not in GRADES:
            errors.append(f"年级错误: {item.get('年级')}")

        # 年龄验证
        if "年龄" in item:
            try:
                age = int(item["年龄"])
                if age < 6 or age > 18:
                    errors.append(f"年龄不在范围[6,18]: {age}")
            except:
                errors.append("年龄非整数")

        # 科目验证
        if "擅长科目" in item and "薄弱科目" in item:
            strong = item["擅长科目"] if isinstance(item["擅长科目"], list) else []
            weak = item["薄弱科目"] if isinstance(item["薄弱科目"], list) else []
            if not strong or not weak:
                errors.append("科目数组为空")
            intersection = set(strong) & set(weak)
            if intersection:
                errors.append(f"科目重叠: {intersection}")

        # 发展阶段验证
        if "发展阶段" in item:
            dev = item["发展阶段"]
            if not isinstance(dev, dict):
                errors.append("发展阶段非对象")
            else:
                required_keys = ["皮亚杰认知发展阶段", "埃里克森心理社会发展阶段", "科尔伯格道德发展阶段"]
                for key in required_keys:
                    if key not in dev:
                        errors.append(f"发展阶段缺失: {key}")

        # 段落字段验证（必须为字符串）
        paragraph_fields = ["人格", "价值观", "社交关系", "创造力", "心理健康"]
        for field in paragraph_fields:
            if field in item:
                val = item[field]
                if not isinstance(val, str) or not val.strip():
                    errors.append(f"{field}非字符串或为空")
                elif "\n\n" in val or re.search(r"^\s*[-•\d]+\.", val, flags=re.MULTILINE):
                    errors.append(f"{field}非单段格式")

        return len(errors) == 0, errors

    def _check_constraint(self, item: Dict[str, Any], mode: str) -> tuple[bool, str]:
        """检查是否符合筛选条件"""
        if mode == "age_15":
            if item.get("年龄") != 15:
                return False, f"年龄不为15: {item.get('年龄')}"
        elif mode == "grade_chu2":
            if item.get("年级") != "初二":
                return False, f"年级不为初二: {item.get('年级')}"

        return True, ""

    def generate_one(self, sid: int, mode: str, target_level: Optional[str] = None,
                     max_retries: int = 3) -> tuple[bool, Optional[Dict[str, Any]], str]:
        """生成一条学生画像"""
        constraint_map = {
            "age_15": "**硬性约束：年龄必须严格等于15岁**",
            "grade_chu2": "**硬性约束：年级必须严格等于初二**",
            "unconstrained": "无特殊约束，生成自然的学生画像"
        }
        constraint_info = constraint_map.get(mode, constraint_map["unconstrained"])

        seed = self._generate_seed()

        for attempt in range(1, max_retries + 1):
            try:
                prompt = self._build_single_prompt(constraint_info, seed, target_level)
                output = self._call_llm(prompt)
                item = self._parse_json_output(output)

                if not item:
                    print(f"[WARN] sid={sid} attempt={attempt} JSON解析失败")
                    continue

                # 设置ID
                item["id"] = sid

                # 验证结构
                ok, errors = self._validate_structure(item)
                if not ok:
                    print(f"[WARN] sid={sid} attempt={attempt} 结构验证失败: {'; '.join(errors)}")
                    continue

                # 验证约束
                ok, err = self._check_constraint(item, mode)
                if not ok:
                    print(f"[WARN] sid={sid} attempt={attempt} 约束验证失败: {err}")
                    continue

                return True, item, ""

            except Exception as e:
                print(f"[ERROR] sid={sid} attempt={attempt} 生成失败: {e}")

        return False, None, f"sid={sid} 在{max_retries}次重试后失败"

# ================== 批量生成与落盘 ==================
def generate_batch(output_dir: str, mode: str, count: int, provider_pool: ProviderPool,
                   max_workers: Optional[int] = None, chunk_size: int = 50):
    """批量生成学生画像"""

    os.makedirs(output_dir, exist_ok=True)

    # 确定并发数
    if max_workers is None:
        clients = getattr(provider_pool, "clients", None) or getattr(provider_pool, "providers", None) or []
        max_workers = max(1, min(count, len(clients) * 2))

    print(f"[INFO] 开始生成，模式={mode}, 数量={count}, 并发={max_workers}")

    generator = SingleShotGenerator(provider_pool)
    write_lock = threading.Lock()
    completed_ids = set()

    # 扫描已完成的ID
    for p in glob.glob(os.path.join(output_dir, "students_chunk_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    completed_ids.add(int(obj.get("id", 0)))
                except:
                    pass

    # 待生成的ID
    pending_ids = [sid for sid in range(1, count + 1) if sid not in completed_ids]

    if not pending_ids:
        print(f"[INFO] 所有{count}条记录已生成完成")
        return

    print(f"[INFO] 已生成 {len(completed_ids)} 条，待生成 {len(pending_ids)} 条")

    # 为了有更好的多样性分布，按轮次均匀分配学术水平
    levels_per_round = []
    remaining = len(pending_ids)
    while remaining > 0:
        # 每轮均匀分配四种水平
        round_levels = [LEVELS[i % 4] for i in range(min(remaining, 4))]
        levels_per_round.extend(round_levels)
        remaining -= len(round_levels)

    # 打乱同时保持一定的均匀性
    random.shuffle(levels_per_round)

    # 构建sid到target_level的映射
    sid_to_level = {sid: levels_per_round[i] for i, sid in enumerate(pending_ids)}

    start_time = time.time()
    success_count = 0
    failed_count = 0

    def _append_record(record: Dict[str, Any]) -> bool:
        """原子性写入一条记录"""
        sid = int(record.get("id", 0))
        chunk_no = (sid - 1) // chunk_size + 1
        path = os.path.join(output_dir, f"students_chunk_{chunk_no}.jsonl")
        line = json.dumps(record, ensure_ascii=False) + "\n"

        with write_lock:
            if sid in completed_ids:
                return False
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            completed_ids.add(sid)
            return True

    # 并发生成
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for sid in pending_ids:
            target_level = sid_to_level.get(sid)
            fut = executor.submit(generator.generate_one, sid, mode, target_level, 3)
            futures[fut] = sid

        for fut in as_completed(futures):
            sid = futures[fut]
            ok, item, err = fut.result()

            if ok and item:
                wrote = _append_record(item)
                if wrote:
                    success_count += 1
                    if success_count % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = success_count / elapsed
                        eta = (len(pending_ids) - success_count) / max(rate, 0.001)
                        print(f"[PROGRESS] 成功: {success_count}/{len(pending_ids)}, "
                              f"速率: {rate:.2f}条/秒, ETA: {eta:.0f}秒")
            else:
                failed_count += 1
                with write_lock:
                    # 记录失败
                    failure_path = os.path.join(output_dir, "failures.jsonl")
                    with open(failure_path, "a", encoding="utf-8") as f:
                        json.dump({"id": sid, "error": err, "timestamp": time.time()},
                                  f, ensure_ascii=False)
                        f.write("\n")

    elapsed = time.time() - start_time
    print(f"\n[COMPLETE] 生成完成")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  平均速率: {success_count/elapsed:.2f}条/秒")
    print(f"  输出目录: {output_dir}")

    # 保存元信息
    meta = {
        "mode": mode,
        "total_n": count,
        "chunk_size": chunk_size,
        "success_count": success_count,
        "failed_count": failed_count,
        "timestamp": time.time()
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ================== 主函数 ==================
def main():
    parser = argparse.ArgumentParser(description="单一大模型生成学生画像基线版本")
    parser.add_argument("--mode", type=str, choices=["age_15", "grade_chu2"],
                       required=True, help="生成模式：age_15=年龄15岁, grade_chu2=年级初二")
    parser.add_argument("--count", type=int, default=5000, help="生成数量")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--keys", type=str, default="secrets/key.txt", help="密钥文件路径")
    parser.add_argument("--max_workers", type=int, default=None, help="最大并发数(默认自动估算)")
    parser.add_argument("--chunk_size", type=int, default=50, help="分片大小")

    args = parser.parse_args()

    # 加载providers
    print(f"[INFO] 加载providers: {args.keys}")
    try:
        providers = load_providers(args.keys)
        pool = ProviderPool(providers)
        print(f"[INFO] 成功加载 {len(providers)} 个providers")
    except Exception as e:
        print(f"[ERROR] 加载providers失败: {e}")
        sys.exit(1)

    # 生成
    generate_batch(args.output, args.mode, args.count, pool,
                   args.max_workers, args.chunk_size)

if __name__ == "__main__":
    main()