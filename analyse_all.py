# -*- coding: utf-8 -*-
"""
evaluate.py — 学生画像 · 离线质量评估（单文件版 · 增强后验版）

仅评估 ./output/merged_baseline.jsonl ，输出写入 ./eval/output/ 目录：
    python analyse_all.py

可选参数：
    --simhash_threshold 3    # 近似检测阈值（汉明距离 <= 阈值 视为相似）
    --topk 100               # 近似对最多保留多少条
    --limit 10000            # 限制评估前N条记录（默认：10000）
    --out_root ./eval        # 评估结果根目录（默认 ./eval）

新增后验信号：
    - 段落长度统计与异常检测（价值观 / 创造力 / 心理健康）
    - [NEW] 段落长度分布直方图（Histogram）
    - 三个长文本维度之间的 Jaccard 相似度（模板化检测）
    - Self-BLEU-2（样本化，用于全局多样性评估，基于字符 n-gram）
    - 学术水平 / 性别 / 年级的全局 KL 散度（对先验分布）
    - 发展阶段(皮亚杰 / 埃里克森 / 科尔伯格) 与年龄的一致性检查
"""

import os, re, json, glob, argparse, hashlib, csv, math, random
from collections import Counter, defaultdict
from typing import List, Dict, Any

# -------------------- 常量 & 规则 --------------------
STRICT_ALLOWED_STRINGS = {
    "高：成绩全校排名前10%",
    "中：成绩全校排名前10%至30%",
    "低：成绩全校排名前30%至50%",
    "差：成绩全校排名后50%"
}
# 学术水平数值化：用于相关系数
LEVEL_NUMERIC = {
    "高：成绩全校排名前10%": 3,
    "中：成绩全校排名前10%至30%": 2,
    "低：成绩全校排名前30%至50%": 1,
    "差：成绩全校排名后50%": 0
}

# 价值观/创造力中常见等级词的极简极性权重
VAL_POLARITY_WEIGHTS = {
    "高": 1.0,
    "较高": 0.8,
    "中上": 0.5,
    "中": 0.0,
    "较低": -0.5,
    "低": -1.0,
}

REQUIRED_KEYS = ["id", "姓名", "年龄", "擅长科目", "薄弱科目", "年级", "人格", "社交关系",
                 "学术水平", "性别", "发展阶段", "代理名", "价值观", "创造力", "心理健康"]

PARAGRAPH_FIELDS = ["价值观", "创造力", "心理健康"]
VAL_DIMS7 = ["道德修养", "身心健康", "法治意识", "社会责任", "政治认同", "文化素养", "家庭观念"]
VAL_LVLS = ["高", "较高", "中", "中上", "较低", "低"]
CRE_DIMS8 = ["流畅性", "新颖性", "灵活性", "可行性", "问题发现", "问题分析", "提出方案", "改善方案"]
PSY_KEYS = ["综合心理状况", "幸福指数", "抑郁风险", "焦虑风险", "信息不足或未见显著症状", "背景", "应对", "支持",
            "家庭", "同伴", "老师", "转折"]

DEFAULT_OUTPUT_ROOT = os.path.join(os.getcwd(), "output")
DEFAULT_EVAL_ROOT = os.path.join(os.getcwd(), "eval")

# 段落长度阈值（可按需要调整）
MIN_PARA_CHAR_LEN = 80
MAX_PARA_CHAR_LEN = 800

# Jaccard 阈值：多维度文本高度相似 → 疑似模板化
TEMPLATE_JACCARD_THRESHOLD = 0.80

# Self-BLEU 抽样配置（防止 O(N^2)）
SELF_BLEU_SAMPLE_SIZE = 2000  # 抽样的 candidate 数量上限
SELF_BLEU_REF_PER_SAMPLE = 20  # 每个 candidate 随机采样多少个参考
SELF_BLEU_MAX_N = 2  # BLEU 的最大 n-gram 维度

# KL 散度的先验目标分布（可按数据/业务调整，需和现实一致性）
TARGET_DIST_ACADEMIC_LEVEL = {
    "高：成绩全校排名前10%": 0.25,
    "中：成绩全校排名前10%至30%": 0.25,
    "低：成绩全校排名前30%至50%": 0.25,
    "差：成绩全校排名后50%": 0.25
}
# 性别先验，这里简单设为 1:1
TARGET_DIST_GENDER = {
    "男": 0.5,
    "女": 0.5
}
# 年级先验：按现在的 12 个年级标签均匀分布
TARGET_DIST_GRADE = {
    "一年级": 1 / 12, "二年级": 1 / 12, "三年级": 1 / 12, "四年级": 1 / 12,
    "五年级": 1 / 12, "六年级": 1 / 12,
    "初一": 1 / 12, "初二": 1 / 12, "初三": 1 / 12,
    "高一": 1 / 12, "高二": 1 / 12, "高三": 1 / 12
}


# -------------------- I/O --------------------
def read_all_records(input_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(input_dir, "merged_students.jsonl")))
    out = []
    for fp in files:
        if not os.path.isfile(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(obj)
                except:
                    pass
    return out


def read_schedule_map(run_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    从 run_dir/schedule.json 读取调度计划，返回:
        { sid(int) : slot(dict) }
    若文件不存在或格式异常，返回空 dict。
    """
    path = os.path.join(run_dir, "schedule.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        return {}
    mp: Dict[int, Dict[str, Any]] = {}
    if isinstance(arr, list):
        for it in arr:
            if not isinstance(it, dict):
                continue
            sid = it.get("sid")
            slot = it.get("slot") or {}
            if isinstance(sid, int) and isinstance(slot, dict):
                mp[sid] = slot
    return mp


# -------------------- 文本与工具 --------------------
def non_empty(v: Any) -> bool:
    if v is None: return False
    if isinstance(v, str): return v.strip() != ""
    if isinstance(v, (list, dict)): return len(v) > 0
    return True


def is_single_paragraph(s: str) -> bool:
    if not isinstance(s, str): return False
    # 禁止双换行或明显的项目符号
    if re.search(r"(\n\s*\n)|(^\s*[-•\d]+\.)", s): return False
    return True


def text_to_ngrams(t: str, n: int = 3):
    t = re.sub(r"\s+", "", str(t))
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))] if t else []


def simhash64(text: str) -> int:
    bits = 64
    v = [0] * bits
    for g in text_to_ngrams(text, 3):
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(bits):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def hamming(a: int, b: int) -> int:
    x = a ^ b
    cnt = 0
    while x:
        x &= x - 1
        cnt += 1
    return cnt


def compute_hamming_stats(simhashes: List[int],
                          max_pairs: int = 50000,
                          seed: int = 42) -> Dict[str, float]:
    """
    SimHash 汉明距离的抽样统计
    """
    n = len(simhashes)
    if n <= 1:
        return {
            "mean": 0.0, "std": 0.0, "mean_normalized": 0.0, "std_normalized": 0.0,
        }

    total_pairs = n * (n - 1) // 2
    M = max_pairs if total_pairs > max_pairs else total_pairs
    if M <= 0:
        return {
            "mean": 0.0, "std": 0.0, "mean_normalized": 0.0, "std_normalized": 0.0,
        }

    random.seed(seed)
    dists = []

    if total_pairs <= max_pairs:
        for i in range(n):
            hi = simhashes[i]
            for j in range(i + 1, n):
                d = hamming(hi, simhashes[j])
                dists.append(d)
    else:
        for _ in range(M):
            i = random.randrange(n)
            j = random.randrange(n - 1)
            if j >= i:
                j += 1
            d = hamming(simhashes[i], simhashes[j])
            dists.append(d)

    if not dists:
        return {
            "mean": 0.0, "std": 0.0, "mean_normalized": 0.0, "std_normalized": 0.0,
        }

    M = len(dists)
    mean = sum(dists) / M
    var = sum((d - mean) * (d - mean) for d in dists) / M
    std = math.sqrt(var) if var > 0 else 0.0

    mean_norm = mean / 64.0
    std_norm = std / 64.0

    return {
        "mean": float(round(mean, 4)),
        "std": float(round(std, 4)),
        "mean_normalized": float(round(mean_norm, 4)),
        "std_normalized": float(round(std_norm, 4)),
    }


def distinct_n(texts: List[str], n: int = 1) -> float:
    ratios = []
    for t in texts:
        tokens = text_to_char_tokens(t)
        if len(tokens) < n:
            continue
        ngrams = set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        denom = max(1, len(tokens) - n + 1)
        ratios.append(len(ngrams) / denom)
    if not ratios:
        return 0.0
    return float(sum(ratios) / len(ratios))


def build_key_text(record: Dict[str, Any]) -> str:
    return "｜".join([
        str(record.get("年级", "")), str(record.get("性别", "")),
        " ".join(record.get("人格", []) if isinstance(record.get("人格"), list) else [str(record.get("人格", ""))]),
        str(record.get("价值观", "")), str(record.get("社交关系", "")),
        str(record.get("创造力", "")), str(record.get("心理健康", ""))
    ])


# --------- 段落长度 / Jaccard / Self-BLEU / KL 等工具函数 ---------
_PUNC_SET = set(" ，。、“”‘’？！：；,.!? \t\r\n｜")


def _clean_chars_for_jaccard(text: str) -> set:
    if not isinstance(text, str):
        text = str(text)
    return {ch for ch in text if ch not in _PUNC_SET}


def text_to_char_tokens(t: str) -> List[str]:
    if not isinstance(t, str):
        t = str(t)
    t = re.sub(r"\s+", "", t)
    return [ch for ch in t if ch not in _PUNC_SET]


# --- 新增：直方图计算工具 ---
def compute_histogram_bins(data: List[int], num_bins: int = 10) -> Dict[str, Any]:
    """
    计算数值列表的直方图分布。
    返回: {
      "labels": ["0-50", "50-100", ...],
      "counts": [10, 25, ...],
      "min": min_val,
      "max": max_val
    }
    """
    if not data:
        return {"labels": [], "counts": [], "min": 0, "max": 0}

    min_v, max_v = min(data), max(data)
    if min_v == max_v:
        return {"labels": [f"{min_v}"], "counts": [len(data)], "min": min_v, "max": max_v}

    # 为了显示美观，我们向上取整 bin_width
    range_span = max_v - min_v
    bin_width = range_span / num_bins

    counts = [0] * num_bins
    labels = []

    # 预生成 labels
    for i in range(num_bins):
        low = min_v + i * bin_width
        high = min_v + (i + 1) * bin_width
        # 最后一个 bin 包含 max_v
        if i == num_bins - 1:
            labels.append(f"{int(low)}-{int(high)}")
        else:
            labels.append(f"{int(low)}-{int(high)}")

    for x in data:
        idx = int((x - min_v) / bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1

    return {"labels": labels, "counts": counts, "min": min_v, "max": max_v}


def draw_ascii_bar(percent: float, width: int = 20) -> str:
    """生成 ASCII 进度条"""
    fill = int(percent * width)
    return "█" * fill + "░" * (width - fill)


# -------------------- Token 计数工具（整条 JSON 画像） --------------------
def get_token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            if not isinstance(text, str):
                text = str(text)
            return len(enc.encode(text))

        return _count, "tiktoken_cl100k_base"
    except Exception:
        import re
        def _count(text: str) -> int:
            if not isinstance(text, str):
                text = str(text)
            text = re.sub(r"\s+", "", text)
            return len(text)

        return _count, "char_length_fallback"


def score_value_text(text: str) -> float:
    if not isinstance(text, str):
        text = str(text)
    total, cnt = 0.0, 0
    for w, weight in VAL_POLARITY_WEIGHTS.items():
        c = len(re.findall(re.escape(w), text))
        if c > 0:
            total += weight * c
            cnt += c
    if cnt == 0:
        return 0.0
    return float(total / cnt)


def score_creativity_text(text: str) -> float:
    if not isinstance(text, str):
        text = str(text)
    total, cnt = 0.0, 0
    for w, weight in VAL_POLARITY_WEIGHTS.items():
        c = len(re.findall(re.escape(w), text))
        if c > 0:
            total += weight * c
            cnt += c
    if cnt == 0:
        return 0.0
    return float(total / cnt)


def score_psychology_text(text: str) -> float:
    if not isinstance(text, str):
        text = str(text)

    score_sum, count = 0.0, 0
    for key in ["综合心理状况", "幸福指数"]:
        pattern = key + r".{0,8}?(高|较高|中上|中|较低|低)"
        for m in re.finditer(pattern, text):
            lvl = m.group(1)
            w = VAL_POLARITY_WEIGHTS.get(lvl, 0.0)
            score_sum += w
            count += 1

    risk_patterns = [
        (r"(抑郁风险|焦虑风险).{0,8}?低风险", -0.2),
        (r"(抑郁风险|焦虑风险).{0,8}?轻度", -0.4),
        (r"(抑郁风险|焦虑风险).{0,8}?中度", -0.7),
        (r"(抑郁风险|焦虑风险).{0,8}?重度", -1.0),
        (r"(抑郁风险|焦虑风险).{0,8}?高", -0.8),
    ]
    for pat, w in risk_patterns:
        for _ in re.finditer(pat, text):
            score_sum += w
            count += 1

    if count == 0:
        return 0.0
    return float(score_sum / count)


def pearson_corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return float(num / (den_x ** 0.5 * den_y ** 0.5))


def jaccard_char(a: str, b: str) -> float:
    sa = _clean_chars_for_jaccard(a)
    sb = _clean_chars_for_jaccard(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def sentence_split(s: str) -> List[str]:
    if not isinstance(s, str):
        s = str(s)
    parts = re.split(r"[。！？!?]+", s)
    return [p for p in parts if p.strip()]


def tokenize_for_bleu(t: str) -> List[str]:
    return text_to_char_tokens(t)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)) if len(tokens) >= n else Counter()


def _bleu_precision(cand: List[str], refs: List[List[str]], n: int) -> float:
    cand_counts = _ngram_counts(cand, n)
    if not cand_counts:
        return 0.0
    max_ref_counts: Dict[tuple, int] = {}
    for r in refs:
        rc = _ngram_counts(r, n)
        for g, c in rc.items():
            if c > max_ref_counts.get(g, 0):
                max_ref_counts[g] = c
    clipped = 0
    for g, c in cand_counts.items():
        clipped += min(c, max_ref_counts.get(g, 0))
    total = sum(cand_counts.values())
    return clipped / total if total > 0 else 0.0


def simple_bleu(cand: List[str], refs: List[List[str]], max_n: int = 2) -> float:
    if not cand or not refs:
        return 0.0
    cand_len = len(cand)
    ref_lens = [len(r) for r in refs if len(r) > 0]
    if not ref_lens:
        return 0.0
    ref_len = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0

    eps = 1e-9
    ps = []
    for n in range(1, max_n + 1):
        p = _bleu_precision(cand, refs, n)
        if p <= 0.0:
            p = eps
        ps.append(p)
    log_p = sum(math.log(p) for p in ps) / max_n
    bleu = bp * math.exp(log_p)
    return float(bleu)


def compute_self_bleu(texts: List[str],
                      sample_size: int = SELF_BLEU_SAMPLE_SIZE,
                      ref_per_sample: int = SELF_BLEU_REF_PER_SAMPLE,
                      max_n: int = SELF_BLEU_MAX_N,
                      seed: int = 42) -> float:
    n = len(texts)
    if n <= 1:
        return 0.0
    random.seed(seed)
    indices = list(range(n))
    if n > sample_size:
        sample_indices = random.sample(indices, sample_size)
    else:
        sample_indices = indices

    token_cache: List[List[str] or None] = [None] * n

    def get_tokens(idx: int) -> List[str]:
        if token_cache[idx] is None:
            token_cache[idx] = tokenize_for_bleu(texts[idx])
        return token_cache[idx]

    scores = []
    for i in sample_indices:
        cand = get_tokens(i)
        if not cand:
            continue
        max_ref_num = min(ref_per_sample, n - 1)
        ref_idx_set = set()
        while len(ref_idx_set) < max_ref_num:
            j = random.randrange(n)
            if j != i:
                ref_idx_set.add(j)
        ref_indices = list(ref_idx_set)
        refs = [get_tokens(j) for j in ref_indices if get_tokens(j)]
        if not refs:
            continue
        bleu = simple_bleu(cand, refs, max_n=max_n)
        scores.append(bleu)

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def kl_divergence_from_counts(counts: Counter, target_dist: Dict[str, float]) -> float:
    keys = set(counts.keys()) | set(target_dist.keys())
    k = len(keys)
    if k == 0:
        return 0.0
    total = sum(counts.values())
    if total == 0:
        return 0.0
    eps = 1e-8
    q_floor = eps

    kl = 0.0
    for key in keys:
        c = counts.get(key, 0)
        p = (c + 1.0) / (total + k)
        q = target_dist.get(key, q_floor)
        if q <= 0.0:
            q = q_floor
        kl += p * math.log(p / q)
    return float(kl)


# -------------------- 代理名：稳健校验 --------------------
_SYL = re.compile(r"^[a-z]+[1-5]$")


def _count_syllables_concat(token: str) -> int:
    if not isinstance(token, str) or not token:
        return 0
    i, n, cnt = 0, len(token), 0
    while i < n:
        j = i
        while j < n and 'a' <= token[j] <= 'z':
            j += 1
        if j == i or j >= n or token[j] not in "12345":
            return 0
        cnt += 1
        i = j + 1
    return cnt


def is_valid_agent_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    if "_" not in name:
        return False

    parts = name.split("_")
    if any(p == "" for p in parts):
        return False

    syllables_per_part = []
    for p in parts:
        syl = _count_syllables_concat(p)
        if syl <= 0:
            return False
        syllables_per_part.append(syl)

    total_syllables = sum(syllables_per_part)
    if total_syllables < 2 or total_syllables > 4:
        return False

    if len(parts) == 2:
        syl_surname = syllables_per_part[0]
        syl_given = syllables_per_part[1]
        if 1 <= syl_surname <= 2 and 1 <= syl_given <= 3 and syl_surname + syl_given <= 4:
            return True
        return False

    syl_surname = syllables_per_part[0]
    syl_given = total_syllables - syl_surname

    if syl_surname < 1 or syl_surname > 2:
        return False
    if syl_given < 1 or syl_given > 3:
        return False

    return True


# -------------------- 规则检查 --------------------
def check_required(record: Dict[str, Any]) -> List[str]:
    miss = []
    for k in REQUIRED_KEYS:
        if k not in record or not non_empty(record[k]):
            miss.append(f"缺失或为空：{k}")
    return miss


def check_academic(record: Dict[str, Any]) -> List[str]:
    issues = []
    lvl = record.get("学术水平", "")
    if lvl not in STRICT_ALLOWED_STRINGS:
        issues.append("学术水平非四选一固定文案")
    good = record.get("擅长科目", [])
    weak = record.get("薄弱科目", [])
    if not isinstance(good, list) or not good:
        issues.append("擅长科目为空或非数组")
    if not isinstance(weak, list) or not weak:
        issues.append("薄弱科目为空或非数组")
    if isinstance(good, list) and isinstance(weak, list):
        if set(good) & set(weak):
            issues.append("擅长/薄弱科目集合相交")
    return issues


def check_agent_name(record: Dict[str, Any]) -> List[str]:
    issues = []
    name = str(record.get("代理名", ""))
    if not is_valid_agent_name(name):
        issues.append("代理名不合规（小写拼音+声调；姓/名用下划线分隔；复姓/名可1–2音节连写或分节）")
    return issues


def check_paragraphs(record: Dict[str, Any]) -> List[str]:
    issues = []
    for f in PARAGRAPH_FIELDS:
        if not is_single_paragraph(record.get(f, "")):
            issues.append(f"{f} 非单段体裁")
    return issues


def check_value_dims(record: Dict[str, Any]) -> List[str]:
    issues = []
    val = str(record.get("价值观", ""))
    if not any(dim in val for dim in VAL_DIMS7):
        issues.append("价值观未见七维显式名词（至少缺大部分）")
    if not any(lv in val for lv in VAL_LVLS):
        issues.append("价值观未见等级词")
    return issues


def check_creativity_dims(record: Dict[str, Any]) -> List[str]:
    issues = []
    cre = str(record.get("创造力", ""))
    if not any(dim in cre for dim in CRE_DIMS8):
        issues.append("创造力未见八维显式名词（至少缺大部分）")
    if ("可行性较低" in cre or "可行性低" in cre) and ("提出方案高" in cre or "提出方案较高" in cre):
        issues.append("创造力一致性：可行性低但‘提出方案’高")
    if not ("雷达" in cre or "总结" in cre):
        issues.append("创造力未见雷达总结提示词")
    return issues


def check_psychology_slots(record: Dict[str, Any]) -> List[str]:
    issues = []
    psy = str(record.get("心理健康", ""))
    if not any(k in psy for k in PSY_KEYS):
        issues.append("心理健康未见关键槽位关键词")
    if "身心健康高" in str(record.get("价值观", "")) or "身心健康较高" in str(record.get("价值观", "")):
        if "重度" in psy:
            issues.append("一致性：价值观身心健康高/较高 ↔ 心理出现重度术语")
    return issues


def check_age_grade_consistency(record: Dict[str, Any]) -> List[str]:
    age = record.get("年龄")
    grade = str(record.get("年级", ""))
    issues = []
    if not isinstance(age, int):
        return ["年龄非整数或缺失"]
    grade_map = {
        "一年级": (7, 8), "二年级": (8, 9),
        "三年级": (9, 10), "四年级": (10, 11),
        "五年级": (10, 11), "六年级": (11, 12),
        "初一": (12, 13), "初二": (13, 14), "初三": (14, 15),
        "高一": (15, 16), "高二": (16, 17), "高三": (17, 18)
    }
    if grade in grade_map:
        lo, hi = grade_map[grade]
        if not (lo - 1 <= age <= hi + 1):
            issues.append(f"年龄-年级非常模(年级:{grade}, 年龄:{age})")
    return issues


def check_dev_stage_consistency(record: Dict[str, Any]) -> List[str]:
    issues = []
    age = record.get("年龄")
    if not isinstance(age, int):
        return issues
    dev = record.get("发展阶段", {})
    if not isinstance(dev, dict):
        return issues

    piaget = str(dev.get("皮亚杰认知发展阶段", ""))
    erikson = str(dev.get("埃里克森心理社会发展阶段", ""))
    kohl = str(dev.get("科尔伯格道德发展阶段", ""))

    if "前运算" in piaget and age >= 10:
        issues.append("发展阶段不一致：前运算阶段通常对应学前及低年级儿童")
    if "具体运算" in piaget and (age < 7 or age > 15):
        issues.append("发展阶段不一致：具体运算阶段通常对应约7–12岁儿童")
    if "形式运算" in piaget and age < 11:
        issues.append("发展阶段不一致：形式运算阶段通常对应11岁以上青少年")

    if "身份与角色混淆" in erikson and age < 11:
        issues.append("发展阶段不一致：身份与角色混淆多见于青春期阶段")
    if "勤奋" in erikson and age > 15:
        issues.append("发展阶段不一致：勤奋与自卑阶段多见于小学阶段儿童")

    if "常规水平" in kohl and age < 9:
        issues.append("发展阶段不一致：道德发展常规水平通常出现在9岁以上")
    if "前习俗" in kohl and age > 13:
        issues.append("发展阶段不一致：前习俗水平多见于儿童早期")

    return issues


# -------------------- 单目录评估 --------------------
def evaluate(records: List[Dict[str, Any]], simhash_th: int = 3, topk: int = 100) -> Dict[str, Any]:
    per_item_results = []
    simhashes = []
    texts = []

    token_counter, token_counter_name = get_token_counter()
    total_tokens = 0
    token_list: List[int] = []

    levels_numeric: List[float] = []
    value_scores: List[float] = []
    creat_scores: List[float] = []
    psy_scores: List[float] = []

    strong_total = weak_total = 0
    strong_hit = weak_hit = 0

    # 段落长度统计：sum / sum_sq / count / 短 / 长
    para_len_stats = {
        f: {"sum_len": 0.0, "sum_sq_len": 0.0, "count": 0, "short": 0, "long": 0}
        for f in PARAGRAPH_FIELDS
    }
    # [NEW] 保留所有段落长度用于画分布图
    all_para_lengths = {f: [] for f in PARAGRAPH_FIELDS}

    sum_j_v_c = sum_j_v_p = sum_j_c_p = 0.0
    jac_count = 0
    template_like_count = 0

    for idx, rec in enumerate(records):
        errs, warns = [], []
        errs += check_required(rec)
        errs += check_academic(rec)
        errs += check_agent_name(rec)
        errs += check_paragraphs(rec)

        warns += check_value_dims(rec)
        warns += check_creativity_dims(rec)
        warns += check_psychology_slots(rec)
        warns += check_age_grade_consistency(rec)
        warns += check_dev_stage_consistency(rec)

        for f in PARAGRAPH_FIELDS:
            text = str(rec.get(f, ""))
            clean = re.sub(r"\s+", "", text)
            char_len = len(clean)

            # 统计
            para_len_stats[f]["sum_len"] += char_len
            para_len_stats[f]["sum_sq_len"] += char_len * char_len
            para_len_stats[f]["count"] += 1
            # [NEW] 记录原始长度
            all_para_lengths[f].append(char_len)

            if char_len < MIN_PARA_CHAR_LEN:
                para_len_stats[f]["short"] += 1
                warns.append(f"{f} 字数明显过短")
            if char_len > MAX_PARA_CHAR_LEN:
                para_len_stats[f]["long"] += 1
                warns.append(f"{f} 字数明显过长")

        v_text = str(rec.get("价值观", ""))
        c_text = str(rec.get("创造力", ""))
        p_text = str(rec.get("心理健康", ""))

        j_v_c = jaccard_char(v_text, c_text)
        j_v_p = jaccard_char(v_text, p_text)
        j_c_p = jaccard_char(c_text, p_text)

        sum_j_v_c += j_v_c
        sum_j_v_p += j_v_p
        sum_j_c_p += j_c_p
        jac_count += 1

        if max(j_v_c, j_v_p, j_c_p) >= TEMPLATE_JACCARD_THRESHOLD:
            template_like_count += 1
            warns.append("多维度文本高度相似（疑似模板化）")

        lvl_str = str(rec.get("学术水平", ""))
        num = LEVEL_NUMERIC.get(lvl_str, None)
        if num is not None:
            vscore = score_value_text(rec.get("价值观", ""))
            cscore = score_creativity_text(rec.get("创造力", ""))
            pscore = score_psychology_text(rec.get("心理健康", ""))
            levels_numeric.append(float(num))
            value_scores.append(float(vscore))
            creat_scores.append(float(cscore))
            psy_scores.append(float(pscore))

        long_text = (
                str(rec.get("价值观", "")) +
                str(rec.get("创造力", "")) +
                str(rec.get("心理健康", "")) +
                str(rec.get("社交关系", ""))
        )
        good = rec.get("擅长科目", [])
        weak = rec.get("薄弱科目", [])

        if isinstance(good, list) and good:
            strong_total += 1
            if any(str(s) in long_text for s in good):
                strong_hit += 1

        if isinstance(weak, list) and weak:
            weak_total += 1
            if any(str(s) in long_text for s in weak):
                weak_hit += 1

        per_item_results.append({
            "id": rec.get("id", idx + 1),
            "姓名": rec.get("姓名", ""),
            "年级": rec.get("年级", ""),
            "性别": rec.get("性别", ""),
            "学术水平": rec.get("学术水平", ""),
            "errors": errs,
            "warnings": warns
        })
        key_text = build_key_text(rec)
        texts.append(key_text)
        simhashes.append(simhash64(key_text))

        json_str = json.dumps(rec, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
        t_i = token_counter(json_str)
        total_tokens += t_i
        token_list.append(t_i)
        per_item_results[-1]["token_full_profile"] = t_i

    if token_list:
        avg_tokens = sum(token_list) / len(token_list)
        min_tokens = min(token_list)
        max_tokens = max(token_list)
    else:
        avg_tokens = 0.0
        min_tokens = 0
        max_tokens = 0

    near_pairs = []
    buckets = defaultdict(list)
    for i, h in enumerate(simhashes):
        bucket_key = h >> 48
        buckets[bucket_key].append(i)
    for idxs in buckets.values():
        m = len(idxs)
        for i in range(m):
            for j in range(i + 1, m):
                a, c = idxs[i], idxs[j]
                if hamming(simhashes[a], simhashes[c]) <= simhash_th:
                    near_pairs.append({
                        "i": a, "j": c,
                        "id_i": per_item_results[a]["id"],
                        "id_j": per_item_results[c]["id"],
                        "姓名_i": per_item_results[a]["姓名"],
                        "姓名_j": per_item_results[c]["姓名"]
                    })
    near_pairs = near_pairs[:topk]
    hamming_stats = compute_hamming_stats(simhashes, max_pairs=50000, seed=42)
    d1 = distinct_n(texts, 1)
    d2 = distinct_n(texts, 2)

    self_bleu2 = compute_self_bleu(
        texts,
        sample_size=SELF_BLEU_SAMPLE_SIZE,
        ref_per_sample=SELF_BLEU_REF_PER_SAMPLE,
        max_n=SELF_BLEU_MAX_N,
        seed=42
    )

    grade_cnt = Counter([str(r.get("年级", "")) for r in records])
    gender_cnt = Counter([str(r.get("性别", "")) for r in records])
    level_cnt = Counter([str(r.get("学术水平", "")) for r in records])

    kl_level = kl_divergence_from_counts(level_cnt, TARGET_DIST_ACADEMIC_LEVEL)
    kl_gender = kl_divergence_from_counts(gender_cnt, TARGET_DIST_GENDER)
    kl_grade = kl_divergence_from_counts(grade_cnt, TARGET_DIST_GRADE)

    err_rate = sum(1 for x in per_item_results if len(x["errors"]) > 0) / max(1, len(per_item_results))
    warn_rate = sum(1 for x in per_item_results if len(x["warnings"]) > 0) / max(1, len(per_item_results))

    para_len_summary = {}
    for f, st in para_len_stats.items():
        cnt = st["count"]
        if cnt <= 0:
            para_len_summary[f] = {
                "mean_len": 0.0,
                "std_len": 0.0,
                "short_ratio": 0.0,
                "long_ratio": 0.0
            }
            continue
        mean = st["sum_len"] / cnt
        var = max(0.0, st["sum_sq_len"] / cnt - mean * mean)
        std = math.sqrt(var)
        para_len_summary[f] = {
            "mean_len": round(mean, 2),
            "std_len": round(std, 2),
            "short_ratio": round(st["short"] / cnt, 4),
            "long_ratio": round(st["long"] / cnt, 4)
        }

    # [NEW] 计算直方图数据
    para_histograms = {}
    for f in PARAGRAPH_FIELDS:
        # 默认分 10 个 bin，或者您可以根据数据量调大
        para_histograms[f] = compute_histogram_bins(all_para_lengths[f], num_bins=12)

    anchor_alignment = {}
    if levels_numeric:
        anchor_alignment["level_vs_values"] = round(
            pearson_corr(levels_numeric, value_scores), 4
        )
        anchor_alignment["level_vs_creativity"] = round(
            pearson_corr(levels_numeric, creat_scores), 4
        )
        anchor_alignment["level_vs_psychology"] = round(
            pearson_corr(levels_numeric, psy_scores), 4
        )
    else:
        anchor_alignment["level_vs_values"] = 0.0
        anchor_alignment["level_vs_creativity"] = 0.0
        anchor_alignment["level_vs_psychology"] = 0.0

    if strong_total > 0:
        strong_ratio = strong_hit / strong_total
    else:
        strong_ratio = 0.0
    if weak_total > 0:
        weak_ratio = weak_hit / weak_total
    else:
        weak_ratio = 0.0

    subject_coverage = {
        "strong_subject_mention_ratio": round(strong_ratio, 4),
        "weak_subject_mention_ratio": round(weak_ratio, 4)
    }

    if jac_count > 0:
        avg_j_v_c = sum_j_v_c / jac_count
        avg_j_v_p = sum_j_v_p / jac_count
        avg_j_c_p = sum_j_c_p / jac_count
    else:
        avg_j_v_c = avg_j_v_p = avg_j_c_p = 0.0

    intra_jaccard_summary = {
        "value_creativity": round(avg_j_v_c, 4),
        "value_psychology": round(avg_j_v_p, 4),
        "creativity_psychology": round(avg_j_c_p, 4),
        "template_like_samples": int(template_like_count)
    }

    n_rec = max(1, len(records))
    miss_rate = sum(1 for r in per_item_results if any("缺失或为空" in e for e in r["errors"])) / n_rec
    para_bad = sum(1 for r in per_item_results if any("非单段体裁" in e for e in r["errors"])) / n_rec
    age_bad = sum(1 for r in per_item_results if
                  any("年龄-年级非常模" in w or "年龄非整数或缺失" in w for w in r["warnings"])) / n_rec
    lvl_bad = sum(1 for r in per_item_results if any("学术水平非四选一" in e for e in r["errors"])) / n_rec
    div_score = max(0.0, min(1.0, 0.5 * (d1 / 0.4) + 0.5 * (d2 / 0.6)))

    score = (
            (1 - miss_rate) * 25 +
            (1 - age_bad) * 25 +
            (1 - para_bad) * 15 +
            div_score * 20 +
            (1 - lvl_bad) * 15
    )

    summary = {
        "count": len(records),
        "error_rate": round(err_rate, 4),
        "warning_rate": round(warn_rate, 4),
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "self_bleu_2": round(self_bleu2, 4),
        "near_duplicate_pairs": len(near_pairs),
        "score_overall_0_100": round(score, 2),
        "distributions": {
            "grade": dict(grade_cnt),
            "gender": dict(gender_cnt),
            "academic_level": dict(level_cnt)
        },
        "paragraph_length_stats": para_len_summary,
        # [NEW] 增加分布数据
        "paragraph_length_dist": para_histograms,
        "intra_jaccard": intra_jaccard_summary,
        "kl_divergence": {
            "academic_level": round(kl_level, 4),
            "gender": round(kl_gender, 4),
            "grade": round(kl_grade, 4)
        },
        "anchor_alignment": anchor_alignment,
        "subject_coverage": subject_coverage,
        "simhash_hamming": hamming_stats,
        "token_usage_full_profile": {
            "total_tokens": int(total_tokens),
            "avg_tokens_per_record": round(avg_tokens, 2),
            "min_tokens": int(min_tokens),
            "max_tokens": int(max_tokens),
            "counter": token_counter_name
        }
    }

    suspicious = sorted(
        per_item_results,
        key=lambda x: (len(x["errors"]), len(x["warnings"])),
        reverse=True
    )[:min(100, len(per_item_results))]

    return {
        "summary": summary,
        "per_item": per_item_results,
        "near_pairs": near_pairs,
        "suspicious": suspicious
    }


# -------------------- 保存 --------------------
def save_one_run(outdir: str, result: Dict[str, Any], run_id: str):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "per_item.jsonl"), "w", encoding="utf-8") as f:
        for r in result["per_item"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(outdir, "near_duplicates.json"), "w", encoding="utf-8") as f:
        json.dump(result["near_pairs"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "suspicious_samples.json"), "w", encoding="utf-8") as f:
        json.dump(result["suspicious"], f, ensure_ascii=False, indent=2)

    rows = [["bucket", "key", "count"]]
    dist = result["summary"]["distributions"]
    for k, v in dist["grade"].items():          rows.append(["grade", k, v])
    for k, v in dist["gender"].items():         rows.append(["gender", k, v])
    for k, v in dist["academic_level"].items(): rows.append(["academic_level", k, v])
    with open(os.path.join(outdir, "buckets.csv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)

    s = result["summary"]
    md = []
    md.append(f"# 学生画像 · 离线评估看板（{run_id}）\n")
    md.append(f"- 总数: **{s['count']}**")
    md.append(f"- 错误率: **{s['error_rate'] * 100:.2f}%**；警告率: **{s['warning_rate'] * 100:.2f}%**")
    md.append(f"- 多样性: Distinct-1 **{s['distinct_1']}** / Distinct-2 **{s['distinct_2']}**")
    md.append(f"- Self-BLEU-2(越低越多样): **{s['self_bleu_2']}**")
    md.append(f"- 近似文本对(SimHash≤阈): **{s['near_duplicate_pairs']}**")
    md.append(f"- 综合评分(0-100): **{s['score_overall_0_100']}**\n")

    md.append("## 覆盖度分布\n")
    for bucket, label in [("grade", "年级"), ("gender", "性别"), ("academic_level", "学术水平")]:
        md.append(f"**{label}**: " + ", ".join([f"{k}:{v}" for k, v in dist[bucket].items()]) + "  ")

    if "token_usage_full_profile" in s:
        tu = s["token_usage_full_profile"]
        md.append("\n## Token 用量统计（整条画像 JSON）\n")
        md.append(f"- 总 token 数: **{tu['total_tokens']}**")
        md.append(f"- 单条画像平均 token 数: **{tu['avg_tokens_per_record']:.2f}**")
        md.append(f"- 最小 / 最大: **{tu['min_tokens']} / {tu['max_tokens']}**")
        md.append(f"- 计数方法: `{tu['counter']}`")

    if "kl_divergence" in s:
        kl = s["kl_divergence"]
        md.append("\n## 分布与先验的 KL 散度\n")
        md.append(f"- 学术水平 KL(P||Q): **{kl['academic_level']}**")
        md.append(f"- 性别 KL(P||Q): **{kl['gender']}**")
        md.append(f"- 年级 KL(P||Q): **{kl['grade']}**")

    # [MODIFIED] 段落长度统计 + 分布图
    if "paragraph_length_stats" in s:
        md.append("\n## 段落长度统计（价值观 / 创造力 / 心理健康）\n")
        pls = s["paragraph_length_stats"]
        pld = s.get("paragraph_length_dist", {})

        for f, st in pls.items():
            md.append(f"### {f}")
            md.append(
                f"- mean_len={st['mean_len']}, std_len={st['std_len']}, "
                f"短比例={st['short_ratio']:.2%}, 长比例={st['long_ratio']:.2%}"
            )
            # 绘制分布图
            if f in pld and pld[f]["counts"]:
                hist = pld[f]
                labels = hist["labels"]
                counts = hist["counts"]
                total = sum(counts) if sum(counts) > 0 else 1
                md.append("\n**分布直方图 (ASCII)**:")
                md.append("```text")
                for lbl, c in zip(labels, counts):
                    bar = draw_ascii_bar(c / total, width=30)
                    md.append(f"{lbl.ljust(10)} | {bar} {c} ({c / total:.1%})")
                md.append("```\n")

    if "simhash_hamming" in s:
        hs = s["simhash_hamming"]
        md.append("\n## SimHash 汉明距离统计\n")
        md.append(
            f"- 平均汉明距离: **{hs['mean']:.2f} / 64** "
            f"(归一化 **{hs['mean_normalized']:.4f}**)"
        )
        md.append(
            f"- 标准差: **{hs['std']:.2f}** "
            f"(归一化 **{hs['std_normalized']:.4f}**)"
        )

    if "intra_jaccard" in s:
        ij = s["intra_jaccard"]
        md.append("\n## 维度间文本相似度（Jaccard，字符级）\n")
        md.append(f"- 价值观 vs 创造力: **{ij['value_creativity']}**")
        md.append(f"- 价值观 vs 心理健康: **{ij['value_psychology']}**")
        md.append(f"- 创造力 vs 心理健康: **{ij['creativity_psychology']}**")
        md.append(f"- 疑似模板化样本数: **{ij['template_like_samples']}**")

    if "match_rate" in s:
        mr = s["match_rate"]
        md.append("\n## 生成计划对齐度（与 schedule.json 对比）\n")
        if "academic_level" in mr:
            md.append(f"- 学术水平匹配率: **{mr['academic_level']:.4f}** ")
        if "grade" in mr:
            md.append(f"- 年级匹配率: **{mr['grade']:.4f}** ")
        if "gender" in mr:
            md.append(f"- 性别匹配率: **{mr['gender']:.4f}** ")

    if "kl_divergence_vs_plan" in s:
        kvp = s["kl_divergence_vs_plan"]
        md.append("\n### 相对于生成计划分布的 KL 散度\n")
        if "academic_level" in kvp:
            md.append(f"- 学术水平 KL(P||Q_plan): **{kvp['academic_level']}**")
        if "grade" in kvp:
            md.append(f"- 年级 KL(P||Q_plan): **{kvp['grade']}**")
        if "gender" in kvp:
            md.append(f"- 性别 KL(P||Q_plan): **{kvp['gender']}**")

    if "anchor_alignment" in s:
        aa = s["anchor_alignment"]
        md.append("\n## 学术水平锚与文本评价的一致性（皮尔逊相关系数）\n")
        md.append(f"- 学术水平 vs 价值观文本: **{aa['level_vs_values']}**")
        md.append(f"- 学术水平 vs 创造力文本: **{aa['level_vs_creativity']}**")
        md.append(f"- 学术水平 vs 心理健康文本: **{aa['level_vs_psychology']}**")

    if "subject_coverage" in s:
        sc = s["subject_coverage"]
        md.append("\n## 学科字段与长文本内容的一致性\n")
        md.append(f"- 擅长科目在长文本中被提及的比例: **{sc['strong_subject_mention_ratio']:.4f}**")
        md.append(f"- 薄弱科目在长文本中被提及的比例: **{sc['weak_subject_mention_ratio']:.4f}**")

    md.append("\n## Top 20 可疑样本（错误/警告最多）\n")
    for r in result["suspicious"][:20]:
        md.append(
            f"- id={r['id']} 姓名={r['姓名']} 年级={r['年级']} 性别={r['性别']}  "
            f"Errors={len(r['errors'])} Warnings={len(r['warnings'])}"
        )
    with open(os.path.join(outdir, "dashboard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# -------------------- 单文件主流程 --------------------
def evaluate_one_run(run_dir: str, out_root: str, simhash_th: int, topk: int, limit: int = None) -> Dict[str, Any]:
    run_id = os.path.basename(os.path.normpath(run_dir))
    records = read_all_records(run_dir)
    if not records:
        raise SystemExit(f"未找到 {os.path.join(run_dir, 'merged_students.jsonl')} 或文件为空，无法评估。")

    # Apply limit if specified
    if limit is not None and limit > 0:
        records = records[:limit]
        print(f"[run]  {run_id}: 读取 {len(records)} 条（限制前{limit}条），开始评估…")
    else:
        print(f"[run]  {run_id}: 读取 {len(records)} 条，开始评估…")

    result = evaluate(records, simhash_th=simhash_th, topk=topk)
    outdir = os.path.join(out_root, run_id)
    schedule_map = read_schedule_map(run_dir)
    if schedule_map:
        from collections import Counter

        s = result["summary"]
        dist = s.get("distributions", {})

        lvl_match_total = lvl_match_ok = 0
        grade_match_total = grade_match_ok = 0
        gender_match_total = gender_match_ok = 0

        plan_level_cnt = Counter()
        plan_grade_cnt = Counter()
        plan_gender_cnt = Counter()

        for rec in records:
            sid = rec.get("id")
            if not isinstance(sid, int):
                continue
            slot = schedule_map.get(sid)
            if not slot:
                continue
            tgt_level = slot.get("目标学术水平")
            tgt_grade = slot.get("年级")
            tgt_gender = slot.get("性别")

            if tgt_level:
                plan_level_cnt[str(tgt_level)] += 1
            if tgt_grade:
                plan_grade_cnt[str(tgt_grade)] += 1
            if tgt_gender:
                plan_gender_cnt[str(tgt_gender)] += 1

            lvl = rec.get("学术水平")
            grade = rec.get("年级")
            gender = rec.get("性别")

            if tgt_level and lvl:
                lvl_match_total += 1
                if str(lvl) == str(tgt_level):
                    lvl_match_ok += 1
            if tgt_grade and grade:
                grade_match_total += 1
                if str(grade) == str(tgt_grade):
                    grade_match_ok += 1
            if tgt_gender and gender:
                gender_match_total += 1
                if str(gender) == str(tgt_gender):
                    gender_match_ok += 1

        match_rate = {}
        if lvl_match_total > 0:
            match_rate["academic_level"] = round(lvl_match_ok / lvl_match_total, 4)
        if grade_match_total > 0:
            match_rate["grade"] = round(grade_match_ok / grade_match_total, 4)
        if gender_match_total > 0:
            match_rate["gender"] = round(gender_match_ok / gender_match_total, 4)
        if match_rate:
            s["match_rate"] = match_rate

        kl_vs_plan = {}
        actual_level_cnt = Counter(dist.get("academic_level", {}))
        actual_grade_cnt = Counter(dist.get("grade", {}))
        actual_gender_cnt = Counter(dist.get("gender", {}))

        if plan_level_cnt:
            total_plan = sum(plan_level_cnt.values())
            q_level = {k: v / total_plan for k, v in plan_level_cnt.items()}
            kl_vs_plan["academic_level"] = round(
                kl_divergence_from_counts(actual_level_cnt, q_level), 4
            )

        if plan_grade_cnt:
            total_plan = sum(plan_grade_cnt.values())
            q_grade = {k: v / total_plan for k, v in plan_grade_cnt.items()}
            kl_vs_plan["grade"] = round(
                kl_divergence_from_counts(actual_grade_cnt, q_grade), 4
            )

        if plan_gender_cnt:
            total_plan = sum(plan_gender_cnt.values())
            q_gender = {k: v / total_plan for k, v in plan_gender_cnt.items()}
            kl_vs_plan["gender"] = round(
                kl_divergence_from_counts(actual_gender_cnt, q_gender), 4
            )

        if kl_vs_plan:
            s["kl_divergence_vs_plan"] = kl_vs_plan

    save_one_run(outdir, result, run_id)
    print(f"[OK] 已完成评估：{run_id}，结果目录：{outdir}")
    s = result["summary"]
    return {
        "run_id": run_id,
        "count": s["count"],
        "error_rate": s["error_rate"],
        "warning_rate": s["warning_rate"],
        "distinct_1": s["distinct_1"],
        "distinct_2": s["distinct_2"],
        "self_bleu_2": s["self_bleu_2"],
        "near_duplicate_pairs": s["near_duplicate_pairs"],
        "score_overall_0_100": s["score_overall_0_100"]
    }


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=DEFAULT_EVAL_ROOT, help="评估结果根目录（默认 ./eval）")
    ap.add_argument("--simhash_threshold", type=int, default=3)
    ap.add_argument("--limit", type=int, default=10000, help="Limit evaluation to first N records (default: 10000)")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    output_root = DEFAULT_OUTPUT_ROOT
    out_root = args.out_root
    sim_th = args.simhash_threshold
    topk = args.topk
    limit = args.limit

    evaluate_one_run(output_root, out_root, sim_th, topk, limit)


if __name__ == "__main__":
    main()