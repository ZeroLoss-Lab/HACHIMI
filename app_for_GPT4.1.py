# -*- coding: utf-8 -*-
"""
app_for_GPT4.1.py — 多智能体 · 实时多轮协作 · 学生画像批量生成（全部字段由智能体经API产出）
在线前置控制：配额分桶调度 + 轻量过滤 + 自相似度阈（SimHash）
后端：Orchestrator + 5 内容Agent + Validator，白板黑板模式 + 多轮协商
依赖：streamlit, requests
接口：所有OpenAI协议接口
!!! 功能要点：
1) 无生成上限：按 50 条/片 分片，支持任意大 N；顶栏“分片进度”，中部“当前片进度”；
2) 无协商轮数上限：轮数 number_input（不设上限）；
3) 暂停/继续：可随时暂停；暂停即作废“当前正在构建”的条目；继续自动找到最后一片并续写；
4) 自动落盘：生成一条就写一条到本地 `output/<run_id>/students_chunk_{i}.jsonl`；50条为一片，自动换新文件；
5) 体裁新标准：价值观/创造力/心理健康 强制“单段连续自然语言”；一致性与合规校验；
6) 学术水平严格“四选一（固定文案）”；代理名允许多音节（姓1–2音节、名1–3音节，每节拼音+1~5声调，用下划线分隔）；
7) QuotaScheduler（年级×性别×优势学科簇）前置采样；轻量过滤；SimHash 去同质化；失败样本落盘；
8) 🖧 实时交互控制台（Prompt/Output/Issues 可视化）；
9) ★ 新增：学术水平分布锚定 + 跨维度“乐观偏置”抑制（价值观/创造力/心理健康随锚自适应，并在轻量过滤中做硬阈校验）。
"""

import json, re, random, math, os, glob, time, hashlib
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional
import streamlit as st
import requests
import heapq
from collections import deque
from providers import load_providers, ProviderPool, set_client_for_thread, get_client_for_thread, ProviderClient
import sys
import traceback
from datetime import datetime



CHUNK_SIZE_DEFAULT = 50
MAX_RETRIES_PER_SLOT = 4
SIMHASH_BITS = 64
SIMHASH_HAMMING_THRESHOLD_DEFAULT = 3  # <=3 视为过近，重采
DEBUG_FULL_RESPONSE = True  # 开启详细调试日志

LEVEL_SET_STRICT = {
    "高": "高：成绩全校排名前10%",
    "中": "中：成绩全校排名前10%至30%",
    "低": "低：成绩全校排名前30%至50%",
    "差": "差：成绩全校排名后50%"
}
STRICT_ALLOWED_STRINGS = set(LEVEL_SET_STRICT.values())
LEVELS = [
    "高：成绩全校排名前10%",
    "中：成绩全校排名前10%至30%",
    "低：成绩全校排名前30%至50%",
    "差：成绩全校排名后50%",
]
LEVEL_ALIAS = {
    "高": LEVELS[0], "中": LEVELS[1], "低": LEVELS[2], "差": LEVELS[3],
    "high": LEVELS[0], "mid": LEVELS[1], "medium": LEVELS[1], "low": LEVELS[2], "poor": LEVELS[3]
}

# ===== 调试日志工具 =====
def _log_debug(msg: str, sid: Optional[int] = None):
    """调试日志函数"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    sid_str = f"[sid={sid}]" if sid is not None else "[GENERAL]"
    print(f"[{timestamp}] DEBUG {sid_str} {msg}", file=sys.stderr, flush=True)

# 代理名校验正则（支持 ≥2 音节；姓1-2节，名1-3节；每节[a-z]++声调1-5；姓与名之间一个下划线）
AGENT_ID_REGEX = r"^(?:[a-z]+[1-5]){1,2}_(?:[a-z]+[1-5]){1,3}$"

GRADES = ["一年级","二年级","三年级","四年级","五年级","六年级","初一","初二","初三","高一","高二","高三"]
GENDERS = ["男","女"]
SUBJ_CLUSTERS = {
    "理科向": ["数学","物理","化学","信息技术"],
    "文社向": ["语文","历史","政治","地理"],
    "艺体向": ["美术","音乐","体育"],
    "外语生物向": ["英语","生物"]
}
from collections import OrderedDict

TARGET_ORDER = [
    "id","姓名","年龄","性别","年级","发展阶段","代理名",
    "擅长科目","薄弱科目","学术水平","人格","价值观",
    "社交关系","创造力","心理健康","_采样约束",
]

def to_canonical_record(rec: dict) -> "OrderedDict[str, any]":
    dev = rec.get("发展阶段", {})
    if not isinstance(dev, dict):
        dev = {"备注": str(dev)}
    def _coerce_list(x): return x if isinstance(x, list) else ([x] if x is not None else [])
    cano = {
        "id": rec.get("id"),
        "姓名": rec.get("姓名"),
        "年龄": rec.get("年龄"),
        "性别": rec.get("性别"),
        "年级": rec.get("年级"),
        "发展阶段": dev,
        "代理名": rec.get("代理名"),
        "擅长科目": _coerce_list(rec.get("擅长科目")),
        "薄弱科目": _coerce_list(rec.get("薄弱科目")),
        "学术水平": rec.get("学术水平"),
        "人格": rec.get("人格"),
        "价值观": rec.get("价值观"),
        "社交关系": rec.get("社交关系"),
        "创造力": rec.get("创造力"),
        "心理健康": rec.get("心理健康"),
        "_采样约束": rec.get("_采样约束"),
    }
    out = OrderedDict()
    for k in TARGET_ORDER:
        if k in cano and cano[k] is not None:
            out[k] = cano[k]
    return out

# ===== 并发与限流支持（AIMD 自适应） =====
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class RateLimiter:
    """
    可变容量令牌桶：capacity = 动态 QPM，窗口 60s，线性补充。
    含 AIMD 自适应：on_success() 累计到 S 次则 +1；on_violation() 乘以 β。
    """
    def __init__(self, qpm_init: int = 6, capacity_max: int = 180):
        self.capacity = max(1, int(qpm_init))
        self.capacity_max = max(self.capacity, int(capacity_max))
        self.tokens = float(self.capacity)
        self.window = 60.0
        self.updated = time.monotonic()
        self.lock = threading.Lock()
        # AIMD
        self.success_streak = 0
        self.additive_step = 1
        self.additive_every = 20
        self.beta = 0.7
        # 统计：成功/违例 60s 滑窗
        self.success_ts = deque()
        self.violation_ts = deque()   # ← 新增
        self.violation_count = 0      # 累计（仅做参考，不参与决策）

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.updated
        if elapsed <= 0: return
        refill = elapsed * (self.capacity / self.window)
        if refill > 0:
            self.tokens = min(self.capacity, self.tokens + refill)
            self.updated = now

    def acquire(self):
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                need = 1.0 - self.tokens
                wait = need * (self.window / max(1e-9, self.capacity))
            time.sleep(min(wait, 1.0))

    def on_success(self):
        with self.lock:
            self.success_streak += 1
            if self.success_streak % self.additive_every == 0:
                self.capacity = min(self.capacity_max, self.capacity + self.additive_step)
                self.tokens = min(self.tokens, self.capacity)
            # 记录成功时间戳（滑窗）
            now = time.monotonic()
            self.success_ts.append(now)

    def on_violation(self):
        with self.lock:
            # 乘性降容
            self.capacity = max(1, int(self.capacity * self.beta))
            self.tokens = min(self.tokens, self.capacity)
            self.success_streak = 0
            # 记录违例：累计 + 60s 窗口
            self.violation_count += 1                   # ← 新增：累计计数
            self.violation_ts.append(time.monotonic())  # ← 新增：时间戳入队

    def get_snapshot(self) -> Dict[str, float]:
        """
        返回当前限流状态快照：
        - capacity_qpm: 目标 QPM（令牌桶容量）
        - observed_qpm: 过去 60 秒成功次数
        - violations_60s: 过去 60 秒违例次数   ← 新增
        - violations_total: 累计违例次数（非决策）
        - tokens: 当前 token 估计
        """
        with self.lock:
            now = time.monotonic()
            cutoff = now - 60.0
            # 清理 60s 外记录
            while self.success_ts and self.success_ts[0] < cutoff:
                self.success_ts.popleft()
            while self.violation_ts and self.violation_ts[0] < cutoff:
                self.violation_ts.popleft()
            observed_qpm = float(len(self.success_ts))
            violations_60s = int(len(self.violation_ts))
            return {
                "capacity_qpm": float(self.capacity),
                "observed_qpm": observed_qpm,
                "violations_60s": violations_60s,     # ← 新增
                "violations_total": int(self.violation_count),
                "tokens": float(self.tokens),
            }

class KeyClient:
    """封装单 key：AIMD 自探测 + 令牌桶限流 + 429/503 退避（含 Retry-After）。"""
    def __init__(self, api_key: str, base_url: str, qpm_init: int = 6, capacity_max: int = 180):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.limiter = RateLimiter(qpm_init=qpm_init, capacity_max=capacity_max)
        # 新增：用于前端展示的匿名标识（不泄露完整 key）
        tail = self.api_key[-6:] if len(self.api_key) >= 6 else self.api_key
        self.label = f"{self.base_url}…{tail}"
        self.lat_ewma = 1.0   # 秒；初始化 1s 当兜底
        self.alpha = 0.2      # EWMA 平滑系数（可调 0.1~0.3）

    def get_qpm_stats(self) -> Dict[str, Any]:
        snap = self.limiter.get_snapshot()
        snap["label"] = self.label
        snap["lat_ewma"] = float(getattr(self, "lat_ewma", 1.0))
        return snap

    def post(self, path: str, payload: dict, timeout: int = 120) -> dict:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        self.limiter.acquire()
        backoff = 0.8
        for attempt in range(1, 5):
            t0 = time.monotonic()  # ← 每次尝试都先打点，避免“未定义”
            r = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
            if r.status_code in (429, 503):
                try:
                    ra = r.headers.get("Retry-After")
                    if ra:
                        time.sleep(float(ra))
                except Exception:
                    pass
                time.sleep(backoff)
                backoff *= 1.8
                self.limiter.on_violation()
                continue
            r.raise_for_status()
            self.limiter.on_success()
            dt = max(1e-3, time.monotonic() - t0)
            self.lat_ewma = self.alpha * dt + (1 - self.alpha) * self.lat_ewma
            return r.json()
        r.raise_for_status()

# 线程局部：让 call_llm 在各线程自动拿到本线程的 client
_TLS = threading.local()
def set_client_for_thread(client: KeyClient): _TLS.client = client
def get_client_for_thread() -> Optional[KeyClient]: return getattr(_TLS, "client", None)

def _estimate_concurrency(pool, kappa: float = 1.3, beta: float = 0.7, cmax: int = 1024) -> int:
    """
    从 provider_pool 的第一个可用 client 估算 C*：
      C* = kappa * (Q/60) * L * penalty
    其中 penalty = beta if violations_60s > 0 else 1
    """
    clients = getattr(pool, "clients", None) or getattr(pool, "providers", None) or []
    if not clients:
        return 1
    c0 = clients[0]
    try:
        stats = c0.get_qpm_stats() if hasattr(c0, "get_qpm_stats") else c0.limiter.get_snapshot()
    except Exception:
        return 1

    Q = float(stats.get("capacity_qpm", 1.0))      # 目标 QPM
    L = float(stats.get("lat_ewma", 1.0))          # 秒
    v60 = int(stats.get("violations_60s", 0))      # ← 新字段：近窗违例

    # 数值稳健性
    L = max(0.2, min(L, 30.0))
    Q = max(1.0, min(Q, 1e6))

    C_star = kappa * (Q / 60.0) * L
    if v60 > 0:
        # 轻抑制：有违例就乘一次 beta；如需更强，可用 beta ** min(v60, 3)
        C_star *= beta

    C_star = int(max(1, min(cmax, round(C_star))))
    return C_star


# ---- streamlit rerun 兼容封装 ----
def _st_rerun():
    try:
        if hasattr(st, 'session_state') and 'completed_sids' in st.session_state:
            completed = len(st.session_state.completed_sids)
            total = st.session_state.total_n
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 触发重绘 -> 进度: {completed}/{total}", file=sys.stderr, flush=True)
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ================== 工具：解析学术水平比例 ==================
def _parse_level_mix(text: str) -> Dict[str, float]:
    """
    解析用户输入的学术水平配比字符串，如：
      高:0.25,中:0.25,低:0.25,差:0.25
      或英文别名：high:0.4,mid:0.3,low:0.2,poor:0.1
    返回严格四选一文案的比例字典；非法或缺失自动均分。
    """
    default = {LEVELS[0]:0.25, LEVELS[1]:0.25, LEVELS[2]:0.25, LEVELS[3]:0.25}
    if not text:
        return default
    try:
        kvs = [x.strip() for x in text.split(",") if x.strip()]
        acc = {}
        for kv in kvs:
            if ":" not in kv:
                continue
            k, v = [t.strip() for t in kv.split(":", 1)]
            k_std = LEVEL_ALIAS.get(k, k)
            if k_std not in STRICT_ALLOWED_STRINGS:
                continue
            acc[k_std] = float(v)
        if not acc: return default
        s = sum(acc.values())
        if s <= 0: return default
        for k in list(acc.keys()):
            acc[k] = acc[k] / s
        for l in LEVELS:
            acc.setdefault(l, 0.0)
        return acc
    except:
        return default

# ================== LLM 调用与解析 ==================
def call_llm(messages: List[Dict[str, Any]], max_tokens=900, temperature=0.95) -> str:
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    client = get_client_for_thread()
    if client is None:
        raise RuntimeError("No provider client bound to this thread. Call set_client_for_thread(client) first.")

    data = client.post("/chat/completions", payload, timeout=120)

    # ==== 完整 response 调试打印（只影响后端日志，不影响 token）====
    if DEBUG_FULL_RESPONSE:
        try:
            # 标一下面是谁的调用（从 messages 里截一小段，非必须）
            head = ""
            if messages:
                head = str(messages[-1].get("content", ""))[:60].replace("\n", " ")
            print("\n" + "=" * 80)
            print(f"[LLM RAW RESPONSE] head={head!r}")
            import json as _json
            print(_json.dumps(data, ensure_ascii=False, indent=2))
            print("=" * 80 + "\n")
        except Exception as e:
            # 打印出错也不影响主流程
            print(f"[LLM RAW RESPONSE] debug print failed: {e}")

    # ==== 真正给 Agent / 白板的仍然只有 content ====
    return data["choices"][0]["message"]["content"]




def try_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return {}
        return {}

def non_empty(v: Any) -> bool:
    if v is None: return False
    if isinstance(v, str): return v.strip() != ""
    if isinstance(v, (list, dict)): return len(v) > 0
    return True

# ================== SimHash（去同质化） ==================
def _text_to_ngrams(t: str, n: int = 3) -> List[str]:
    t = re.sub(r"\s+", "", t)
    return [t[i:i+n] for i in range(max(0, len(t)-n+1))] if t else []

def _simhash64(text: str) -> int:
    v = [0]*SIMHASH_BITS
    for g in _text_to_ngrams(text, 3):
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        for i in range(SIMHASH_BITS):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(SIMHASH_BITS):
        if v[i] >= 0:
            out |= (1 << i)
    return out

def _hamming(a: int, b: int) -> int:
    x = a ^ b
    cnt = 0
    while x:
        x &= x-1
        cnt += 1
    return cnt

class SimilarityGate:
    def __init__(self, threshold: int = SIMHASH_HAMMING_THRESHOLD_DEFAULT):
        self.threshold = threshold
        self.pool: List[int] = []
        self.lock = threading.Lock()

    def too_similar(self, text: str) -> bool:
        if not text: return False
        h = _simhash64(text)
        with self.lock:
            for prev in self.pool:
                if _hamming(h, prev) <= self.threshold:
                    return True
        return False

    def accept(self, text: str):
        if not text: return
        with self.lock:
            self.pool.append(_simhash64(text))

    def try_accept(self, text: str) -> bool:
        """原子：相似性检查 + 接纳。"""
        if not text: return True
        h = _simhash64(text)
        with self.lock:
            for prev in self.pool:
                if _hamming(h, prev) <= self.threshold:
                    return False
            self.pool.append(h)
            return True


# ================== 轻量过滤（体裁/正则/显式命中 + 乐观偏置抑制） ==================
AGENT_PARAGRAPH_FIELDS = ["价值观","创造力","心理健康"]
VAL_DIMS7 = ["道德修养","身心健康","法治意识","社会责任","政治认同","文化素养","家庭观念"]
LVL_WORDS = ["高","较高","中上","中","较低","低"]
CRE_DIMS8 = ["流畅性","新颖性","灵活性","可行性","问题发现","问题分析","提出方案","改善方案"]
PSY_KEYS  = ["综合心理状况","幸福指数","抑郁风险","焦虑风险"]





def _is_single_paragraph(s: str) -> bool:
    if not isinstance(s, str): return False
    if re.search(r"(\n\s*\n)|(^\s*[-•\d]+\.)", s): return False
    return True

def _has_any(s: str, kws: List[str]) -> bool:
    return any(kw in s for kw in kws)

def _count_levels(text: str) -> Dict[str, int]:
    cnt = {k:0 for k in LVL_WORDS}
    for k in LVL_WORDS:
        cnt[k] = len(re.findall(re.escape(k), text))
    return cnt

def _count_lowish(text: str) -> int:
    # 统计“中/较低/低”（不含“中上”）
    n_mid = len(re.findall(r"(?<!中)中(?!上)", text))
    n_low = len(re.findall(r"较低|低", text))
    return n_mid + n_low

def _extract_dim_levels(text: str, dims: List[str]) -> Dict[str, str]:
    """
    近似抽取每个维度的等级词（正则启发式），用于八维/七维粗校验。
    """
    res = {}
    for d in dims:
        # 维度名后若干字符内的等级词
        m = re.search(d + r".{0,12}?(高|较高|中上|中(?!上)|较低|低)", text)
        if m: res[d] = m.group(1)
    return res

def _light_filter(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons = []
    # 1) 必填键
    for k in ["姓名","年龄","性别","年级","人格","擅长科目","薄弱科目","学术水平","价值观","创造力","心理健康","代理名","发展阶段","社交关系"]:
        if k not in item or not non_empty(item[k]):
            reasons.append(f"缺字段或为空：{k}")

    # 2) 学术水平+代理名+段落体裁
    if item.get("学术水平") not in STRICT_ALLOWED_STRINGS:
        reasons.append("学术水平非四选一固定文案")
    if not re.match(AGENT_ID_REGEX, str(item.get("代理名",""))):
        reasons.append("代理名不合规（需拼音分节+声调数字；姓1-2节，名1-3节）")
    for f in AGENT_PARAGRAPH_FIELDS:
        if not _is_single_paragraph(item.get(f,"")):
            reasons.append(f"{f} 非单段体裁")

    # 3) 价值观：七维&等级词 + 乐观偏置抑制
    val = item.get("价值观","")
    if not _has_any(val, VAL_DIMS7): reasons.append("价值观未见七维显式名词（至少缺大部分）")
    if not _has_any(val, LVL_WORDS): reasons.append("价值观未见等级词")
    # 锚定驱动的下调要求
    target = item.get("_采样约束",{}).get("目标学术水平") if isinstance(item.get("_采样约束"), dict) else None
    lowish_need = 0
    if target in [LEVELS[1]]:      # 中
        lowish_need = 1
    elif target in [LEVELS[2]]:    # 低
        lowish_need = 2
    elif target in [LEVELS[3]]:    # 差
        lowish_need = 3
    if lowish_need>0 and _count_lowish(val) < lowish_need:
        reasons.append(f"价值观等级分布过高（锚={target or '无'}）：需要≥{lowish_need}处“中/较低/低”")

    # 4) 创造力：八维 + 雷达 + 乐观偏置抑制 + 内部一致性
    cre = item.get("创造力","")
    if not _has_any(cre, CRE_DIMS8): reasons.append("创造力未见八维显式名词（至少缺大部分）")
    if "雷达" not in cre and "总结" not in cre: reasons.append("创造力未见雷达总结提示词")
    dimlv = _extract_dim_levels(cre, CRE_DIMS8)
    # 至少 N 个维度为“中及以下”（不含“中上”）
    lowish_cre = sum(1 for v in dimlv.values() if v in ["中","较低","低"])
    need = 0
    if target in [LEVELS[1]]:  # 中
        need = 2
    elif target in [LEVELS[2]]:  # 低
        need = 3
    elif target in [LEVELS[3]]:  # 差
        need = 4
    if need>0 and lowish_cre < need:
        reasons.append(f"创造力八维整体偏高（锚={target or '无'}）：要求≥{need}个维度为“中及以下”，当前={lowish_cre}")
    # 原有一致性：可行性低→提出方案≤中
    if ("可行性" in dimlv and dimlv.get("可行性") in ["较低","低"]) and \
       ("提出方案" in dimlv and dimlv.get("提出方案") in ["高","较高","中上"]):
        reasons.append("创造力一致性：可行性低但‘提出方案’高")

    # 5) 心理健康：关键槽位 + 乐观偏置抑制（四槽位至少出现中/较低/低；风险不能全低于‘中’的反面）
    psy = item.get("心理健康","")
    if not _has_any(psy, ["综合心理状况","幸福指数","抑郁风险","焦虑风险","信息不足或未见显著症状","背景","应对","支持","家庭","同伴","老师"]):
        reasons.append("心理健康未见核心槽位关键词")
    # 粗抽四槽位等级
    psy_map = {}
    for k in PSY_KEYS:
        m = re.search(k + r".{0,12}?(高|较高|中上|中(?!上)|较低|低|轻度|中度|重度|低风险)", psy)
        if m: psy_map[k] = m.group(1)
    if target in [LEVELS[2], LEVELS[3]]:  # 低/差
        # 要求：综合心理状况/幸福指数 至少一个为“中及以下”；抑郁/焦虑风险不得都写成“低/低风险”
        cnt_mid_or_low = sum(1 for k in ["综合心理状况","幸福指数"] if psy_map.get(k) in ["中","较低","低"])
        if cnt_mid_or_low < 1:
            reasons.append("心理健康与锚不符：综合心理状况/幸福指数至少1处需“中或较低/低”")
        risk_lowish = 0
        for k in ["抑郁风险","焦虑风险"]:
            v = psy_map.get(k, "")
            if any(x in v for x in ["轻度","中度"]):  # 允许轻/中
                risk_lowish += 1
        # 若两项都显式“低/低风险”，在‘低/差’锚下不合理
        if "抑郁风险" in psy_map and "焦虑风险" in psy_map:
            both_low = all(("低" in psy_map[k] or "低风险" in psy_map[k]) for k in ["抑郁风险","焦虑风险"])
            if both_low:
                reasons.append("心理健康与锚不符：抑郁/焦虑风险不应双双为‘低/低风险’")
    return len(reasons) == 0, reasons

# ================== 协作基石：白板与讨论（支持IO日志） ==================
REQUIRED_KEYS = ["id","姓名","年龄","擅长科目","薄弱科目","年级","人格","社交关系",
                 "学术水平","性别","发展阶段","代理名","价值观","创造力","心理健康"]

class Whiteboard:
    def __init__(self, sid: int, sampling_hint: Optional[Dict[str,Any]] = None):
        self.facts: Dict[str, Any] = {"id": sid}
        if sampling_hint:
            self.facts["_采样约束"] = sampling_hint
        self.discussion: List[Dict[str, str]] = []

    def read(self) -> Dict[str, Any]:
        return deepcopy(self.facts)

    def write(self, patch: Dict[str, Any]):
        for k, v in patch.items():
            self.facts[k] = v

    def log(self, speaker: str, content: str):
        # 仍然保留完整日志，供前端“交互控制台”展示
        self.discussion.append({"speaker": speaker, "content": content})

    def serialize_for_agent(self) -> str:
        """
        ★ 关键改动：给 LLM 的只剩当前草稿 draft，
        不再带上累积 discussion，从而把每轮 prompt
        的长度控制在 O(|facts|) 而不是 O(轮数×指令).
        """
        return json.dumps({"draft": self.facts}, ensure_ascii=False)
        # 如果你想更极端地精简，也可以：
        # return json.dumps(self.facts, ensure_ascii=False)
        # 但为了兼容原来“draft”这个键，建议先用上面那一行。

# ================== 基础提示词：统一协议 ==================
AGENT_PREAMBLE = """你是一个与其他智能体协作的“学生画像”生产成员。我们使用“公共白板”共享草稿与讨论。
规则（必须遵守）：
- 所有输出必须是 **合法 JSON 对象**，且只包含你负责的键。
- 不得引用模板句式；用自然中文；避免空话套话；避免与白板草稿自相矛盾。
- 若被要求修订，只改你负责的键；不留空；保证与其它字段逻辑一致。
- 姓名需为中文；数字与百分位请用中文语境书写（如“前10%”）。
- 不要输出任何多余说明文字。只输出 JSON。
- 如白板中存在“_采样约束”，请严格遵循其中的“年级”“性别”“优势学科偏向”“目标学术水平”等要求；若发生冲突，以采样约束为准并保持整体一致性。
"""

RESP_FIELDS = {
    "学籍与发展阶段": ["姓名","年龄","性别","年级","发展阶段","代理名"],
    "学业画像": ["擅长科目","薄弱科目","学术水平"],
    "人格与价值观": ["人格","价值观"],
    "社交与创造力": ["社交关系","创造力"],
    "身心健康": ["心理健康"]
}

def _pack_prompt(instruction: str, wb: Whiteboard) -> str:
    return f"【INSTRUCTION】\n{instruction}\n\n【WHITEBOARD】\n{wb.serialize_for_agent()}"

# ================== 各 Agent（含自适应锚指引 + IO日志） ==================
def agent_scholar(wb: Whiteboard, seed: str, mode: str="propose") -> Dict[str,Any]:
    sampling = wb.read().get("_采样约束", {})
    hint = ""
    if sampling:
        hint = f"\n采样约束（遵循）：年级={sampling.get('年级','未指定')}，性别={sampling.get('性别','未指定')}，目标学术水平={sampling.get('目标学术水平','无')}。"
    instruction = f"""{AGENT_PREAMBLE}{hint}
你负责键：{RESP_FIELDS["学籍与发展阶段"]}
任务模式：{mode}
多样性种子：{seed}

生成与约束（必须）：
- 年龄 6~18；年龄**必须是一个阿拉伯数字**，年级与年龄匹配（允许±1年跳级/留级但需与其他段落一致）；
- 发展阶段对象必须含三键：皮亚杰认知发展阶段、埃里克森心理社会发展阶段、科尔伯格道德发展阶段；
- 代理名格式（**多音节支持**）：姓 1~2 音节、名 1~3 音节；每个音节为“拼音小写+声调数字(1-5)”；姓与名之间用下划线；示例：
  - 单姓单名：zhang1_shuang3
  - 单姓双名：li1_huan4ying1
  - 复姓双名：ou3yang2_ming2hao3
仅输出 JSON。
"""
    messages = [
        {"role":"developer","content":instruction},
        {"role":"user","content":f"公共白板：\n{wb.serialize_for_agent()}\n请仅输出你负责的 JSON。"}
    ]
    # 只记录 instruction，本身不再嵌入 WHITEBOARD，避免指数膨胀
    wb.log("学籍与发展阶段→prompt", instruction)
    out = call_llm(messages, max_tokens=700, temperature=0.98)
    wb.log("学籍与发展阶段←output", out)

    return try_json(out)

def agent_academic(wb: Whiteboard, seed: str, mode: str="propose") -> Dict[str,Any]:
    sampling = wb.read().get("_采样约束", {})
    prefer = sampling.get("优势学科偏向")
    prefer_str = f"请优先使“擅长科目”覆盖该簇中的至少1门：{prefer}。" if prefer else ""
    target_level = sampling.get("目标学术水平")
    target_line = f"【强约束】本样本的“学术水平”必须严格等于：{target_level}；不得改为其它档位。" if target_level else "（无目标锚）"

    instruction = f"""{AGENT_PREAMBLE}
你负责键：{RESP_FIELDS["学业画像"]}
任务模式：{mode}
多样性种子：{seed}

要求（必须）：
- “擅长科目”与“薄弱科目”均为非空数组，且两者**集合不相交**；
- “学术水平”**严格四选一，且字符串必须完全等于以下之一**：
  1) "高：成绩全校排名前10%"
  2) "中：成绩全校排名前10%至30%"
  3) "低：成绩全校排名前30%至50%"
  4) "差：成绩全校排名后50%"
- {prefer_str}
- {target_line}

仅输出 JSON（只含“擅长科目”“薄弱科目”“学术水平”三个键）。
"""
    messages = [
        {"role":"developer","content":instruction},
        {"role":"user","content":f"公共白板：\n{wb.serialize_for_agent()}\n请仅输出你负责的 JSON。"}
    ]
    wb.log("学业画像→prompt", instruction)
    out = call_llm(messages, max_tokens=600, temperature=0.9)
    wb.log("学业画像←output", out)
    data = try_json(out)

    # 兜底归一 + 强制对齐目标锚（如存在）
    if isinstance(data, dict):
        lvl = data.get("学术水平")
        if isinstance(lvl, str):
            for k, v in LEVEL_SET_STRICT.items():
                if lvl.startswith(k) or k in lvl:
                    data["学术水平"] = v; break
        if target_level and data.get("学术水平") != target_level:
            data["学术水平"] = target_level
    return data if isinstance(data, dict) else {}

def agent_values(wb: Whiteboard, seed: str, mode: str="propose") -> Dict[str,Any]:
    sampling = wb.read().get("_采样约束", {})
    target = sampling.get("目标学术水平")
    adapt = ""
    if target in [LEVELS[1], LEVELS[2], LEVELS[3]]:
        adapt = ("- 【随学术锚自适应】当目标为“中/低/差”时，七维中的等级词应呈**不均衡但包含若干“中/较低/低”**，"
                 "避免全高/较高；并给出与之匹配的背景化根据（如学习习惯/反馈/社团表现等）。")
    instruction = f"""{AGENT_PREAMBLE}
你负责键：{RESP_FIELDS["人格与价值观"]}
任务模式：{mode}
多样性种子：{seed}

输出体裁（强约束）：单段连续自然语言；**覆盖七维并有等级词**（道德修养、身心健康、法治意识、社会责任、政治认同、文化素养、家庭观念）；给出背景化依据。
{adapt}
仅输出 JSON。
"""
    messages = [
        {"role":"developer","content":instruction},
        {"role":"user","content":f"公共白板：\n{wb.serialize_for_agent()}\n请仅输出你负责的 JSON。"}
    ]
    wb.log("人格与价值观→prompt", instruction)
    out = call_llm(messages, max_tokens=900, temperature=1.0)
    wb.log("人格与价值观←output", out)

    return try_json(out)

def agent_social_creative(wb: Whiteboard, seed: str, mode: str="propose") -> Dict[str,Any]:
    sampling = wb.read().get("_采样约束", {})
    target = sampling.get("目标学术水平")
    adapt = ""
    if target in [LEVELS[1], LEVELS[2], LEVELS[3]]:
        adapt = ("- 【随学术锚自适应】当目标为“中/低/差”时，八维等级分布**必须包含若干“中/较低/低”**（至少2/3/4个维度），"
                 "并保持可行性与提出方案的一致性；末尾雷达总结据此概括强弱。")
    instruction = f"""{AGENT_PREAMBLE}
你负责键：{RESP_FIELDS["社交与创造力"]}
任务模式：{mode}
多样性种子：{seed}

社交关系：单段（160~260字），背景→关键事件→影响；不得换行/条列。
创造力：单段；**八维（流畅性/新颖性/灵活性/可行性/问题发现/问题分析/提出方案/改善方案 各有等级词）+ 雷达总结**；八维不得全同档；若“可行性”较低/低，则“提出方案”不高于中等。
{adapt}
仅输出 JSON。
"""
    messages = [
        {"role":"developer","content":instruction},
        {"role":"user","content":f"公共白板：\n{wb.serialize_for_agent()}\n请仅输出你负责的 JSON。"}
    ]
    wb.log("社交与创造力→prompt", instruction)
    out = call_llm(messages, max_tokens=1100, temperature=1.02)
    wb.log("社交与创造力←output", out)

    return try_json(out)

def agent_health(wb: Whiteboard, seed: str, mode: str="propose") -> Dict[str,Any]:
    sampling = wb.read().get("_采样约束", {})
    target = sampling.get("目标学术水平")
    adapt = ""
    if target in [LEVELS[2], LEVELS[3]]:
        adapt = ("- 【随学术锚自适应】当目标为“低/差”时，“综合心理状况/幸福指数”中至少一项宜为“中或较低/低”；"
                 "抑郁/焦虑风险避免双双‘低’；仍须保持**非诊断化**与“可支持、可改善”的教育语境。")
    instruction = f"""{AGENT_PREAMBLE}
你负责键：{RESP_FIELDS["身心健康"]}
任务模式：{mode}
多样性种子：{seed}

心理健康：单段；依次内嵌 概述→性格特征(≥2)→综合心理状况/幸福指数/抑郁风险/焦虑风险→心理疾病（如无写“信息不足或未见显著症状”）→背景故事→支撑与应对；行文要求非诊断化；与价值观“身心健康”一致。
{adapt}
仅输出 JSON。
"""
    messages = [
        {"role":"developer","content":instruction},
        {"role":"user","content":f"公共白板：\n{wb.serialize_for_agent()}\n请仅输出你负责的 JSON。"}
    ]
    wb.log("身心健康→prompt", instruction)
    out = call_llm(messages, max_tokens=1100, temperature=0.96)
    wb.log("身心健康←output", out)

    return try_json(out)

def agent_validator_fast(wb: Whiteboard, seed: str) -> Dict[str, Any]:
    """
    Validator（Fast）：使用 Deep 规则的一个小子集，
    只覆盖关键结构、体裁和显著矛盾，一次性给出 issues/final_ready。
    """
    instruction = f"""你是“Validator”智能体。请严格审校并给出结构化修订任务。
{AGENT_PREAMBLE}
你只输出 JSON，键为 issues 与 final_ready，不要输出多余文字。
若未发现问题，应返回 issues=[] 且 final_ready=true。
仅按照下列 F1–F4 规则审查，不额外引入其他判断或主观标准：

F1 结构：id、姓名、年龄、性别、年级、学术水平、人格、社交关系、
擅长科目、薄弱科目、发展阶段、代理名、价值观、创造力、心理健康
都必须存在且非空。年龄为 6–20 岁整数；性别∈{{"男","女"}}；
擅长/薄弱科目为非空数组且集合不交；代理名符合正则：^(?:[a-z]+[1-5]){1,2}_(?:[a-z]+[1-5]){1,3}$

F2 体裁：价值观、创造力、心理健康三段必须是单段自然语言，
不得出现空行或条列符号，例如“\\n\\n”“- ”“• ”“1.” 等。

F3 年龄–年级–发展阶段：年龄与年级需落在常见小学/初中/高中区间内，
允许 ±1 岁浮动；发展阶段标签与年龄不得构成明显不可能的组合。

F4 跨段一致性：标记显著矛盾情形，例如：
价值观中把身心状态描述为稳定健康，而心理健康段又写为
长期抑郁、功能受损；或者社交段写长期回避、
不与同伴互动，而人格/社交又写外向、人缘极佳。"""

    messages = [
        {"role": "developer", "content": instruction},
        {"role": "user", "content": f"公共白板：\n{wb.serialize_for_agent()}\n请只输出 JSON。"}
    ]

    wb.log("Validator-Fast→prompt", instruction)
    out = call_llm(messages, max_tokens=450, temperature=0.1)
    wb.log("Validator-Fast←output", out)

    data = try_json(out)
    if not isinstance(data, dict):
        # 兜底：解析失败就认为本轮不通过，交给后续流程处理
        data = {
            "issues": [{
                "code": "F_SYS",
                "desc": "Validator-Fast 未返回合法 JSON。",
                "owner": "学籍与发展阶段",
                "fields": ["姓名"],
                "hint": "按指令，仅返回含 issues 与 final_ready 的 JSON。"
            }],
            "final_ready": False
        }

    # 若没有任何 issue，则显式标记为 final_ready=true
    if isinstance(data.get("issues"), list) and len(data["issues"]) == 0:
        data.setdefault("final_ready", True)

    return data


def agent_validator_deep(wb: Whiteboard, seed: str) -> Dict[str, Any]:
    """
        Validator-Deep：昂贵的深度一致性+规范性检查。
        在 Validator-Fast 之后、且仅在必要时才调用，用于抓细粒度的一致性问题。
        """
    instruction = f"""你是“Validator”智能体。请严格审校并给出**结构化修订任务**。
    {AGENT_PREAMBLE}
    你只输出 JSON，键为 issues 与 final_ready。不要输出多余文字。
    """
    rules = f"""规则参考（必须）：
- R1 年龄↔年级常模：6-7一年级；7-8二；8-9三；9-10四；10-11五；11-12六；12-13初一；13-14初二；14-15初三；15-16高一；16-17高二；17-18高三（允许±1年内偏差）。
- R2 发展阶段与年龄：~12岁以下多为“具体运算”；~12岁以上“形式运算”。埃里克森：6-12勤奋vs自卑；12-18身份vs角色混乱；科尔伯格：~10前习俗、~10-15习俗、≥15可向后习俗过渡。
- R3 科目集合不交叉、且均非空。
- R4 创造力八维等级需有起伏，避免全部相同；若“可行性”较低/低，则“提出方案”不高于中等。
- R5 价值观积极稳健时，心理段落不得出现严重功能受损或重度临床术语。
- R6 代理名正则：^(?:[a-z]+[1-5]){1,2}_(?:[a-z]+[1-5]){1,3}$
- R7 所有必填键不可为空：id, 姓名, 年龄, 擅长科目, 薄弱科目, 年级, 人格, 社交关系, 学术水平, 性别, 发展阶段, 代理名, 价值观, 创造力, 心理健康。
- R8 价值观：必须覆盖七维（道德修养/身心健康/法治意识/社会责任/政治认同/文化素养/家庭观念），每维含可识别等级词；允许自然顺序与自由句法，但需可定位。
- R9 创造力：必须含 概述 + 八维（流畅性/新颖性/灵活性/可行性/问题发现/问题分析/提出方案/改善方案，逐维有等级词与简短依据）+ 雷达总结。
- R10 心理健康：必须含 概述 + 性格特征(≥2点) + 三维度（综合心理状况/幸福指数/抑郁风险与焦虑风险） + 心理疾病（若无写“信息不足或未见显著症状”，若有写“诊断或倾向/功能影响/当前支持与处理”） + 背景故事 + 支撑与应对。
- R11 一致性：
    · 若价值观“身心健康”为“较高/高”，则心理“综合心理状况”≥中等，且“抑郁/焦虑风险”≤中度；如涉及疾病，需“已管理、功能基本稳定”；
    · 家庭观念较高与独立性不冲突，应呈现“互动支持、边界清晰”；
    · 价值观/社交/学业叙事互相支撑，不得矛盾（如社交回避 vs 频繁协作）。
- R12 非诊断化语言：避免“重度抑郁/双相/用药/住院”等重临床表述；允许“倾向/轻度/节点性/阶段性/可管理/建议咨询”等。
- R13 可读性与避免模板：内容应自然连贯，拒绝流水账与机械复述；若“等级词”缺失或维度缺失，提出修订。
- R14 段落化体裁：价值观/创造力/心理健康必须为**单段连续自然语言**，不得使用列表、编号、项目符号或多段换行；如检测到“\\n\\n”、“1.”、“- ”、“• ”等条列痕迹，应要求对应Owner重写为单段。
- R15 若“学术水平”不在允许集合，必须要求“学业画像”Owner重写并替换为**严格四选一固定文案**。
输出：issues: [{{code, desc, owner, fields, hint}}], final_ready: bool
"""
    messages = [
        {"role": "developer", "content": instruction},
        {"role": "user", "content": f"公共白板：\n{wb.serialize_for_agent()}\n{rules}\n请输出 JSON。"}
    ]
    # 同样只把“指令+规则”记到 discussion，不再嵌入 WHITEBOARD
    wb.log("Validator-deep→prompt", instruction + "\n\n" + rules)
    out = call_llm(messages, max_tokens=1100, temperature=0.2)
    wb.log("Validator-deep←output", out)

    data = try_json(out)
    # 本地兜底（学术水平、代理名、与目标锚一致性）
    try:
        lvl = wb.read().get("学术水平", "")
        if lvl not in STRICT_ALLOWED_STRINGS:
            issues = data.get("issues", []) if isinstance(data, dict) else []
            issues.append({
                "code":"R14",
                "desc":"学术水平未严格匹配允许集合。",
                "owner":"学业画像",
                "fields":["学术水平"],
                "hint":"替换为四选一固定文案：'高：成绩全校排名前10%' / '中：成绩全校排名前10%至30%' / '低：成绩全校排名前30%至50%' / '差：成绩全校排名后50%'"})
            data = {"issues": issues, "final_ready": False}
        agent_id = wb.read().get("代理名", "")
        if not re.match(AGENT_ID_REGEX, str(agent_id)):
            issues = data.get("issues", []) if isinstance(data, dict) else []
            issues.append({
                "code":"R6",
                "desc":"代理名不合规（应为多音节拼音+声调数字，姓1-2节，名1-3节，姓与名用下划线分隔）。",
                "owner":"学籍与发展阶段",
                "fields":["代理名"],
                "hint":"示例：zhang1_shuang3 / li1_huan4ying1 / ou3yang2_ming2hao3"})
            data = {"issues": issues, "final_ready": False}
        sampling = wb.read().get("_采样约束", {})
        target_level = sampling.get("目标学术水平")
        if target_level and lvl != target_level:
            issues = data.get("issues", []) if isinstance(data, dict) else []
            issues.append({
                "code":"R14-anchored",
                "desc":f"与采样目标学术水平不一致（期望：{target_level}，实际：{lvl}）。",
                "owner":"学业画像",
                "fields":["学术水平"],
                "hint":f"将“学术水平”改为目标档位：{target_level}；其余字段做轻微一致性修订。"})
            data = {"issues": issues, "final_ready": False}
    except Exception:
        pass
    try:
        wb.log("Validator(issues)", json.dumps(data.get("issues", []), ensure_ascii=False))
    except Exception:
        wb.log("Validator(issues)", "[]")
    return data if data else {"issues":[{"code":"SYS","desc":"解析失败，请各Agent自检并重述其负责字段。","owner":"学籍与发展阶段","fields":["姓名"],"hint":"重新完整给出。"}],"final_ready":False}

# ================== Orchestrator ==================
class Orchestrator:
    def __init__(self, max_rounds:int=3):
        self.max_rounds = max_rounds
        self.used_names: set = set()

    def _seed(self) -> str:
        return f"SEED-{random.randrange(10**16,10**17-1)}"

    def _merge_and_log(self, wb: Whiteboard, patch: Dict[str, Any], agent_name: str):
        wb.write(patch)
        wb.log(agent_name+"(合并)", json.dumps(patch, ensure_ascii=False))

    def run_one(self, sid: int, sampling_hint: Optional[Dict[str, Any]] = None,
                client: Optional[KeyClient] = None) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        # 在本线程设置 client（关键一步）
        if client is not None:
            set_client_for_thread(client)
        wb = Whiteboard(sid, sampling_hint=sampling_hint)
        wb.log("System", f"以下姓名已被使用（请避免重复）：{list(self.used_names)}")
        seed = self._seed()

        self._merge_and_log(wb, agent_scholar(wb, seed, "propose"), "学籍与发展阶段")
        name_now = wb.read().get("姓名")
        if name_now: self.used_names.add(name_now)

        self._merge_and_log(wb, agent_academic(wb, seed, "propose"), "学业画像")
        self._merge_and_log(wb, agent_values(wb, seed, "propose"), "人格与价值观")
        self._merge_and_log(wb, agent_social_creative(wb, seed, "propose"), "社交与创造力")
        self._merge_and_log(wb, agent_health(wb, seed, "propose"), "身心健康")

        # ===== 第一步：Fast Validator（只跑一轮，结构/体裁/显而易见错误） =====
        v_fast = agent_validator_fast(wb, self._seed())
        issues_fast = v_fast.get("issues", []) if isinstance(v_fast, dict) else []
        final_ready_fast = bool(v_fast.get("final_ready", False)) if isinstance(v_fast, dict) else False

        if final_ready_fast and not issues_fast:
            # Fast 认为“结构 + 轻量一致性”已满足，直接通过，不再调用 Deep
            wb.log("Orchestrator", "Validator-Fast 通过 ✅，跳过深度 Validator。")
        else:
            # Fast 检出问题（或自身解析失败），交给 Deep 做精细修订
            wb.log("Orchestrator", f"Validator-Fast 检出 {len(issues_fast)} 个问题，进入深度 Validator。")

            # ===== 第二步：Deep Validator 多轮修订（最多 max_rounds 轮） =====
            for r in range(1, self.max_rounds + 1):
                v = agent_validator_deep(wb, self._seed())
                issues = v.get("issues", [])
                final_ready = bool(v.get("final_ready", False))

                if final_ready and not issues:
                    wb.log("Orchestrator", f"深度 Validator 第{r}轮通过 ✅，停止修订。")
                    break

                wb.log("Orchestrator", f"深度 Validator 第{r}轮：收到 {len(issues)} 个修订任务。")
                owners = {
                    "学籍与发展阶段": agent_scholar,
                    "学业画像": agent_academic,
                    "人格与价值观": agent_values,
                    "社交与创造力": agent_social_creative,
                    "身心健康": agent_health
                }
                wb.log("Validator-Deep(issues)", json.dumps(issues, ensure_ascii=False))

                touched = set()
                for it in issues:
                    owner = it.get("owner")
                    if owner in owners and owner not in touched:
                        patched = owners[owner](wb, self._seed(), "revise")
                        self._merge_and_log(wb, patched, owner + "(revise)")
                        touched.add(owner)

        final = wb.read()
        final["id"] = int(sid)  # 强制将最终成品 id 对齐到调度 sid

        missing = [k for k in REQUIRED_KEYS if not non_empty(final.get(k))]
        if missing:
            wb.log("Orchestrator", f"最终补齐：缺失 {missing}")
            for k in missing:
                owner = next((owner for owner,keys in RESP_FIELDS.items() if k in keys), None)
                if owner == "学籍与发展阶段":
                    self._merge_and_log(wb, agent_scholar(wb, self._seed(), "revise"), "学籍与发展阶段(revise-final)")
                elif owner == "学业画像":
                    self._merge_and_log(wb, agent_academic(wb, self._seed(), "revise"), "学业画像(revise-final)")
                elif owner == "人格与价值观":
                    self._merge_and_log(wb, agent_values(wb, self._seed(), "revise"), "人格与价值观(revise-final)")
                elif owner == "社交与创造力":
                    self._merge_and_log(wb, agent_social_creative(wb, self._seed(), "revise"), "社交与创造力(revise-final)")
                elif owner == "身心健康":
                    self._merge_and_log(wb, agent_health(wb, self._seed(), "revise"), "身心健康(revise-final)")
            final = wb.read()

        for k in REQUIRED_KEYS:
            if not non_empty(final.get(k)):
                raise RuntimeError(f"字段仍为空：{k}")

        lvl = final.get("学术水平", "")
        if lvl not in STRICT_ALLOWED_STRINGS:
            raise RuntimeError("学术水平不符合严格四选一标准，请重试。")

        # 注意：在落盘前保留 _采样约束 用于在线过滤的锚参考。落盘时你也可以选择 pop 掉。
        return final, wb.discussion

def _worker_process_one(orch: Orchestrator, sid: int, slot: Dict[str, Any],
                        client: KeyClient, sim_gate: SimilarityGate,
                        max_retries: int = MAX_RETRIES_PER_SLOT) -> Tuple[bool, Optional[Dict[str,Any]], Optional[List[Dict[str,str]]], str]:
    """
    返回：(accepted, item, dialog, err_msg)
    - 在工作线程内完成：run_one + 轻量过滤 + 相似度原子接纳（try_accept）
    - 若不过关则在本线程内按 MAX_RETRIES_PER_SLOT 重试
    """
    _log_debug(f"开始处理 slot={slot}", sid=sid)
    set_client_for_thread(client)
    last_dialog = None
    last_err = ""
    for attempt in range(1, max_retries+1):
        _log_debug(f"尝试第 {attempt}/{max_retries} 次", sid=sid)
        try:
            _log_debug(f"调用 orch.run_one", sid=sid)
            item, dialog = orch.run_one(sid, sampling_hint=slot, client=client)
            _log_debug(f"orch.run_one 返回", sid=sid)

            # 轻量过滤
            ok, reasons = _light_filter(item)
            if not ok:
                last_err = f"轻量过滤不通过：{'; '.join(reasons)}"
                _log_debug(f"轻量过滤不通过：{reasons}", sid=sid)
                continue

            # 原子相似度接纳（并发安全）
            key_text = "｜".join([
                str(item.get("年级","")), str(item.get("性别","")),
                " ".join(item.get("人格",[]) if isinstance(item.get("人格"), list) else [str(item.get("人格",""))]),
                str(item.get("价值观","")), str(item.get("社交关系","")),
                str(item.get("创造力","")), str(item.get("心理健康",""))
            ])
            if not sim_gate.try_accept(key_text):
                last_err = f"与已有样本过近（SimHash≤{sim_gate.threshold}）"
                _log_debug(f"SimHash 相似度检查不通过（阈值={sim_gate.threshold}）", sid=sid)
                continue

            # 通过
            _log_debug(f"生成成功，准备返回", sid=sid)
            return True, item, dialog, ""
        except Exception as e:
            last_err = f"尝试{attempt}/{max_retries}失败：{e}"
            _log_debug(f"异常：{last_err}\n{traceback.format_exc()}", sid=sid)
        finally:
            last_dialog = dialog if 'dialog' in locals() else last_dialog
    _log_debug(f"所有重试失败，返回 False", sid=sid)
    return False, None, last_dialog, last_err

# ================== QuotaScheduler：按比例生成“目标学术水平” ==================
def _default_quota(n_total: int) -> List[Dict[str,Any]]:
    slots = []
    triplets = [(g,s,c) for g in GRADES for s in GENDERS for c in SUBJ_CLUSTERS.keys()]
    for i in range(n_total):
        g, s, c = triplets[i % len(triplets)]
        slots.append({"年级": g, "性别": s, "优势学科偏向": SUBJ_CLUSTERS[c]})
    random.shuffle(slots)
    return slots

def _cycle_levels_by_mix(n_total: int, mix: Dict[str, float]) -> List[str]:
    import math, random
    targets = []
    alloc = {k: int(round(mix.get(k,0.0) * n_total)) for k in STRICT_ALLOWED_STRINGS}
    diff = n_total - sum(alloc.values())
    if diff != 0:
        order = sorted(STRICT_ALLOWED_STRINGS, key=lambda k: mix.get(k,0.0), reverse=True)
        i = 0
        while diff != 0:
            k = order[i % len(order)]
            if diff > 0:
                alloc[k] += 1; diff -= 1
            else:
                if alloc[k] > 0:
                    alloc[k] -= 1; diff += 1
            i += 1
    for k, c in alloc.items():
        targets.extend([k] * max(0, c))
    random.shuffle(targets)
    if len(targets) < n_total:
        pad = list(STRICT_ALLOWED_STRINGS)
        while len(targets) < n_total:
            targets.append(random.choice(pad))
    return targets[:n_total]

class QuotaScheduler:
    def __init__(self, n_total: int, user_quota_json: Optional[str] = None, level_mix: Optional[Dict[str,float]] = None):
        if user_quota_json:
            try:
                arr = json.loads(user_quota_json)
                assert isinstance(arr, list) and all(isinstance(x, dict) for x in arr)
                self.slots = arr
            except Exception:
                self.slots = _default_quota(n_total)
        else:
            self.slots = _default_quota(n_total)
        mix = level_mix or {LEVELS[0]:0.25, LEVELS[1]:0.25, LEVELS[2]:0.25, LEVELS[3]:0.25}
        targets = _cycle_levels_by_mix(n_total, mix)
        for i, t in enumerate(targets):
            self.slots[i]["目标学术水平"] = t
        self.idx = 0
        self.total = n_total
    def has_next(self) -> bool:
        return self.idx < self.total
    def next_slot(self) -> Dict[str,Any]:
        if not self.has_next(): return {}
        slot = self.slots[self.idx]; self.idx += 1; return slot

# ================== 本地落盘（自动 JSONL） ==================
from collections import deque

def _save_schedule(run_dir: str, schedule: List[Dict[str, Any]]):
    path = os.path.join(run_dir, "schedule.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)

def _load_schedule(run_dir: str) -> Optional[List[Dict[str,Any]]]:
    path = os.path.join(run_dir, "schedule.json")
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _scan_completed_ids(run_dir: str) -> set:
    done = set()
    for p in glob.glob(os.path.join(run_dir, "students_chunk_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    sid = int(obj.get("id", 0))
                    if sid > 0: done.add(sid)
                except:
                    pass
    return done


# >>> 新增：meta.json 读写（持久化 total_n / chunk_size）
def _save_meta(run_dir: str, total_n: int, chunk_size: int):
    """
    将关键参数持久化，确保 resume 时与启动时保持一致：
    - total_n: 目标样本总数
    - chunk_size: 分片大小
    """
    path = os.path.join(run_dir, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"total_n": int(total_n), "chunk_size": int(chunk_size)}, f, ensure_ascii=False)

def _load_meta(run_dir: str) -> Optional[Dict[str, int]]:
    """
    读取 meta.json；不存在则返回 None
    """
    path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
# 并发写锁（启动时赋值）
# st.session_state.write_lock = threading.Lock()

def _append_record_by_sid(run_dir: str, chunk_size: int, record: Dict[str, Any], write_lock: threading.Lock) -> bool:
    """
    幂等写：同一 sid 仅写一次；在同一锁域内完成“检查→写入→标记→推进 next_sid”。
    返回 True 表示本次确实写入；False 表示该 sid 先前已写过（本次跳过）。
    """
    # 强制对齐到目标结构（第二种）
    record = to_canonical_record(record)

    sid = int(record.get("id", 0))
    if sid <= 0:
        raise RuntimeError("record 缺少合法 id")

    chunk_no = (sid - 1) // chunk_size + 1
    path = _chunk_path(run_dir, chunk_no)
    line = json.dumps(record, ensure_ascii=False) + "\n"

    with write_lock:
        # —— 幂等栅栏：防二次写入（并发/重试/恢复均有效）——
        if sid in st.session_state.completed_sids:
            return False

        # 真正写盘
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

        # 标记完成 & 推进 next_sid（保持单调）
        st.session_state.completed_sids.add(sid)
        while (st.session_state.next_sid in st.session_state.completed_sids
               and st.session_state.next_sid <= st.session_state.total_n):
            st.session_state.next_sid += 1
    return True



def _ensure_dirs():
    base = os.path.join(os.getcwd(), "output")
    if not os.path.exists(base): os.makedirs(base, exist_ok=True)
    return base

def _init_run_dir():
    base = _ensure_dirs()
    run_id = f"run_{int(time.time())}"
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)


def _resolve_run_dir(user_path: str) -> str:
    """
    将用户输入目录解析为绝对路径；相对路径自动挂到 ./output 下。
    """
    p = (user_path or "").strip()
    if not p:
        raise ValueError("输出目录不能为空")
    if os.path.isabs(p):
        return p
    base = _ensure_dirs()  # ./output
    return os.path.join(base, p)

def _chunk_path(run_dir: str, chunk_no: int) -> str:
    return os.path.join(run_dir, f"students_chunk_{chunk_no}.jsonl")

def _append_record(run_dir: str, chunk_no: int, record: Dict[str, Any]):
    path = _chunk_path(run_dir, chunk_no)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _append_failure(run_dir: str, failure: Dict[str, Any]):
    path = os.path.join(run_dir, "failures.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(failure, ensure_ascii=False) + "\n")

def _count_lines(path: str) -> int:
    if not os.path.exists(path): return 0
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f: cnt += 1
    return cnt

def _recover_progress_from_disk(run_dir: str, chunk_size: int) -> Tuple[int,int,int]:
    files = sorted(glob.glob(os.path.join(run_dir, "students_chunk_*.jsonl")))
    if not files: return 1, 0, 1
    def _num(p):
        m = re.search(r"students_chunk_(\d+)\.jsonl$", p)
        return int(m.group(1)) if m else 0
    files.sort(key=_num)
    last = files[-1]; last_no = _num(last)
    done_in_last = _count_lines(last)
    chunk_idx = last_no; in_chunk_idx = done_in_last
    global_idx = (chunk_idx-1)*chunk_size + in_chunk_idx + 1
    if in_chunk_idx >= chunk_size: chunk_idx += 1; in_chunk_idx = 0
    return chunk_idx, in_chunk_idx, global_idx

def _load_chunk_preview(run_dir: str, chunk_idx: int, max_items: int = 6) -> List[Dict[str, Any]]:
    path = _chunk_path(run_dir, chunk_idx)
    if not os.path.exists(path): return []
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: lines.append(json.loads(line))
            except: pass
    return lines[-max_items:]

# ================== UI ==================
st.set_page_config(page_title="多智能体画像生成（前置控制+交互控制台+分布锚定）", page_icon="🧩", layout="wide")


st.title("🧩 学生画像 · 多智能体实时协作（前置控制 + 交互控制台 + 分布锚定）")

# ========= 首屏初始化：要求先绑定落盘目录 =========
import os, re
import streamlit as st

SAFE_BASE = "./output"  # 可改为你允许的根目录前缀

def _ensure_base_dir():
    os.makedirs(SAFE_BASE, exist_ok=True)

def _is_safe_subpath(base: str, sub: str) -> bool:
    # 防止 .. 注入；仅允许相对 SAFE_BASE 的子路径（或以 SAFE_BASE 为前缀的绝对路径）
    base = os.path.abspath(base)
    sub_abs = os.path.abspath(sub)
    return sub_abs.startswith(base + os.sep) or (sub_abs == base)

def _resolve_dir(user_input: str) -> str:
    p = (user_input or "").strip()
    if not p:
        raise ValueError("输出目录不能为空")
    # 相对路径 -> SAFE_BASE/p
    if not os.path.isabs(p):
        _ensure_base_dir()
        tgt = os.path.join(SAFE_BASE, p)
    else:
        tgt = p
    # 目录安全校验（可按需放宽）
    if not _is_safe_subpath(SAFE_BASE, tgt):
        raise ValueError("目录不在允许的根路径内，或包含非法路径片段。")
    return tgt

# 只初始化一次
if "run_dir" not in st.session_state:
    st.session_state.run_dir = ""

# 若未绑定目录，则展示“首屏向导”并阻塞后续逻辑
if not st.session_state.run_dir:
    st.title("🧩 首次使用：请先绑定落盘目录")
    with st.form("setup_form"):
        # 支持：1) 选择已有子目录；2) 输入新目录名；3) 也可输入绝对路径（若你允许）
        _ensure_base_dir()
        existing = sorted(
            [d for d in os.listdir(SAFE_BASE) if os.path.isdir(os.path.join(SAFE_BASE, d))]
        )
        col1, col2 = st.columns([1,2])
        with col1:
            pick = st.selectbox("选择已有子目录（可选）", ["（不选）"] + existing, index=0)
        with col2:
            new_sub = st.text_input("或输入新的子目录名（推荐）", placeholder="my_run_2025_10_31")
        abs_opt  = st.text_input("或直接输入绝对路径（高级）", placeholder="/data/runs/exp1")
        ok = st.form_submit_button("绑定目录并进入应用 ▶️")

    if ok:
        try:
            # 解析优先级：绝对路径 > 选择已有 > 新目录名
            raw = abs_opt or (pick if pick != "（不选）" else new_sub)
            run_dir = _resolve_dir(raw)
            os.makedirs(run_dir, exist_ok=True)
            # 基本可写性检查
            testfile = os.path.join(run_dir, ".write_test")
            with open(testfile, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(testfile)

            st.session_state.run_dir = run_dir
            st.success(f"已绑定：{run_dir}")
            st.rerun()
        except Exception as e:
            st.error(f"绑定目录失败：{e}")
    # 阻止后续主界面渲染（直到绑定成功）
    st.stop()

from providers import load_providers, ProviderPool, ProviderClient
# —— 会话态自举：在任何 UI 使用前确保键存在 ——
if "provider_pool" not in st.session_state:
    st.session_state.provider_pool = None
if "keys_path" not in st.session_state:
    # 默认：脚本所在目录 /secrets/key.txt 的绝对路径
    import os
    st.session_state.keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets", "key.txt")



with st.sidebar:
    st.subheader("在线前置控制")
    # 你原有的控件（示例）
    simhash_th = st.number_input("相似度阈（SimHash汉明距离，≤视为过近需重生）", 0, 16, SIMHASH_HAMMING_THRESHOLD_DEFAULT)
    user_quota_json = st.text_area("自定义配额JSON（可选）", placeholder='[{"年级":"初一","性别":"女","优势学科偏向":["英语","生物"]}]')
    show_console = st.toggle("显示交互控制台（Prompt/Output/Issues）", value=True)
    level_mix_text = st.text_input(
        "学术水平比例（高/中/低/差），如：高:0.25,中:0.25,低:0.25,差:0.25",
        value="高:0.25,中:0.25,低:0.25,差:0.25"
    )

    st.caption("密钥与厂商信息仅保存在后端 secrets/key.txt，不在前端展示；支持多厂商混合调用（OpenAI 范式）。")

    # —— 安全读取：若键不存在则给默认值（双保险） ——
    default_keys_path = st.session_state.get("keys_path", "secrets/key.txt")
    keys_path = st.text_input("密钥文件路径（后端可读）", value=st.session_state.keys_path)
    reload_btn = st.button("重载提供者 🔄")
    per_step_m = st.number_input("每步并发生成条数（建议 ≤ 提供者数×1）", min_value=1, value=10, step=1)

# —— 提供者池初始化 / 热重载（首次渲染自动加载；点击按钮或路径变化时强制重载） ——
def _ensure_provider_pool(force: bool = False):
    need = force or (st.session_state.provider_pool is None)
    if need:
        try:
            st.session_state.provider_pool = ProviderPool(load_providers(keys_path))
            st.session_state.keys_path = keys_path
            st.success("提供者加载成功。")
        except Exception as e:
            st.session_state.provider_pool = None
            st.error(f"加载提供者失败：{e}")
            st.stop()

if reload_btn or (keys_path != st.session_state.keys_path):
    _ensure_provider_pool(force=True)
else:
    _ensure_provider_pool(force=False)



with st.expander("说明", expanded=False):
    st.markdown("""
- **配额分桶调度**、**轻量过滤**、**SimHash 去同质化**；不过关即重采；  
- 自动落盘：`output/<run_id>/students_chunk_{i}.jsonl`；失败样本 `failures.jsonl`；  
- 学术水平四选一（固定文案）；代理名：姓 1–2 音节、名 1–3 音节，每节“拼音+1~5 声调”，下划线分隔。  
- **分布锚定**：侧边栏控制“高/中/低/差”比例；`agent_academic` 强制输出、`Validator` 兜底一致；  
- **乐观偏置抑制**：价值观/创造力/心理健康随锚自适应，轻量过滤中要求“中/较低/低”的**最小计数**（目标为“中/低/差”时生效）。  
""")

left, right = st.columns([1,3])
with left:
    n = st.number_input("生成数量（无限制）", min_value=1, value=1000000, step=1)
    rounds = st.number_input("最大协商轮数（无限制）", min_value=1, value=3, step=1)
    chunk_size = CHUNK_SIZE_DEFAULT
    start_btn = st.button("开始/继续生成 ▶️", type="primary")
    pause_btn = st.button("暂停生成 ⏸")

    # 只读展示当前绑定目录；支持一键“更换”
    st.text_input("当前落盘目录（只读）", value=st.session_state.run_dir, disabled=True)
    if st.button("更换落盘目录"):
        # 置空并回到首屏向导
        st.session_state.run_dir = ""
        st.rerun()

# ----------- 状态初始化 -----------
if "running" not in st.session_state: st.session_state.running = False
if "paused" not in st.session_state: st.session_state.paused = False
if "total_n" not in st.session_state: st.session_state.total_n = 0
if "max_rounds" not in st.session_state: st.session_state.max_rounds = 3
if "chunk_size" not in st.session_state: st.session_state.chunk_size = CHUNK_SIZE_DEFAULT
if "chunks_total" not in st.session_state: st.session_state.chunks_total = 0
if "chunk_idx" not in st.session_state: st.session_state.chunk_idx = 1
if "in_chunk_idx" not in st.session_state: st.session_state.in_chunk_idx = 0
if "global_idx" not in st.session_state: st.session_state.global_idx = 1
if "orch" not in st.session_state: st.session_state.orch = None
if "run_id" not in st.session_state: st.session_state.run_id = None
if "run_dir" not in st.session_state: st.session_state.run_dir = ""
if "last_item" not in st.session_state: st.session_state.last_item = None
if "last_dialog" not in st.session_state: st.session_state.last_dialog = []
if "last_error" not in st.session_state: st.session_state.last_error = None
if "quota" not in st.session_state: st.session_state.quota = None
if "sim_gate" not in st.session_state: st.session_state.sim_gate = SimilarityGate(threshold=simhash_th)
if "level_mix" not in st.session_state: st.session_state.level_mix = _parse_level_mix(level_mix_text)
if "schedule" not in st.session_state: st.session_state.schedule = []           # [{'sid':1,'slot':{...}}, ...]
if "sid2slot" not in st.session_state: st.session_state.sid2slot = {}
if "pending" not in st.session_state: st.session_state.pending = []  # 最小堆（heapq）存 sid

if "completed_sids" not in st.session_state: st.session_state.completed_sids = set()
if "next_sid" not in st.session_state: st.session_state.next_sid = 1
if "write_lock" not in st.session_state: st.session_state.write_lock = threading.Lock()


# ----------- 控制按钮 -----------
if start_btn:
    # 0) 只读会话内已绑定目录；若空则报错（理论上不会出现）
    run_dir = st.session_state.run_dir
    if not run_dir:
        st.error("尚未绑定落盘目录，请先在首屏完成绑定。")
        st.stop()
    os.makedirs(run_dir, exist_ok=True)

    # 1) 载入既有计划/元信息；扫描已完成
    schedule = _load_schedule(run_dir)
    meta     = _load_meta(run_dir)
    done     = _scan_completed_ids(run_dir)
    max_done_sid = max(done) if done else 0

    # 2) Orchestrator 与在线控制（每次开始/继续都可安全重建）
    st.session_state.orch = Orchestrator(max_rounds=int(rounds))
    st.session_state.level_mix = _parse_level_mix(level_mix_text)
    st.session_state.sim_gate = SimilarityGate(threshold=simhash_th)
    st.session_state.last_item = None
    st.session_state.last_dialog = []
    st.session_state.last_error = None


    # 3) 若 schedule 不存在：按“max(n, 已有最大 sid)”新建；否则沿用/扩展
    if schedule is None:
        total_n = max(int(n), max_done_sid)  # 确保能“接上”已存在的最大 sid
        qs = QuotaScheduler(total_n, user_quota_json=user_quota_json, level_mix=st.session_state.level_mix)
        schedule = [{"sid": sid, "slot": qs.next_slot() if qs.has_next() else {}} for sid in range(1, total_n+1)]
        _save_schedule(run_dir, schedule)

        # 新建 meta（chunk_size 使用当前 UI；可按需固定为 CHUNK_SIZE_DEFAULT）
        st.session_state.total_n = total_n
        st.session_state.chunk_size = int(CHUNK_SIZE_DEFAULT)  # 或 int(chunk_size)
        _save_meta(run_dir, st.session_state.total_n, st.session_state.chunk_size)
    else:
        # 已有计划：严格沿用原 meta；若缺失 meta，则以 schedule 长度回填
        st.session_state.sid2slot = {it["sid"]: it["slot"] for it in schedule}
        if meta:
            st.session_state.total_n  = int(meta.get("total_n", len(schedule)))
            st.session_state.chunk_size = int(meta.get("chunk_size", st.session_state.chunk_size))
        else:
            st.session_state.total_n  = len(schedule)

        # 若用户这次输入的 n > 已记录 total_n，则**扩展**计划并更新 meta
        if int(n) > st.session_state.total_n:
            add = int(n) - st.session_state.total_n
            qs  = QuotaScheduler(add, user_quota_json=user_quota_json, level_mix=st.session_state.level_mix)
            ext = [{"sid": sid, "slot": qs.next_slot() if qs.has_next() else {}}
                   for sid in range(st.session_state.total_n+1, int(n)+1)]
            schedule.extend(ext)
            _save_schedule(run_dir, schedule)
            st.session_state.sid2slot.update({it["sid"]: it["slot"] for it in ext})
            st.session_state.total_n = int(n)
            _save_meta(run_dir, st.session_state.total_n, st.session_state.chunk_size)

    # 4) 建立 sid→slot 映射（新建时需要；已有时上面已设）
    if "sid2slot" not in st.session_state or not st.session_state.sid2slot:
        st.session_state.sid2slot = {it["sid"]: it["slot"] for it in schedule}

    # 5) 计算总片数并恢复未完成堆
    st.session_state.chunks_total = math.ceil(st.session_state.total_n / st.session_state.chunk_size)
    st.session_state.completed_sids = set(done)
    st.session_state.pending = [sid for sid in range(1, st.session_state.total_n + 1) if sid not in done]
    heapq.heapify(st.session_state.pending)

    # 6) next_sid = 最小未完成
    ns = 1
    while ns in st.session_state.completed_sids and ns <= st.session_state.total_n:
        ns += 1
    st.session_state.next_sid = ns

    # 7) 置为“运行/非暂停”
    st.session_state.running = True
    st.session_state.paused = False

    st.success(f"已绑定输出目录：{run_dir}；进度 {len(done)}/{st.session_state.total_n}，最小缺口 sid={st.session_state.next_sid}")
    _st_rerun()



if pause_btn and st.session_state.running:
    st.session_state.paused = True

# if resume_btn and st.session_state.running:
#     st.session_state.paused = False
#     if st.session_state.run_dir:
#         # 恢复 schedule（若没有则就地重建：按当前配额生成，但为了稳定期望 schedule.json 应该存在）
#         schedule = _load_schedule(st.session_state.run_dir)
#         if schedule is None:
#             # 回退方案：基于当前 QuotaScheduler 再配一次
#             tmp_quota = QuotaScheduler(st.session_state.total_n, user_quota_json=user_quota_json, level_mix=st.session_state.level_mix)
#             schedule = [{"sid": sid, "slot": tmp_quota.next_slot() if tmp_quota.has_next() else {}} for sid in range(1, st.session_state.total_n+1)]
#             _save_schedule(st.session_state.run_dir, schedule)
#         st.session_state.schedule = schedule
#
#         st.session_state.sid2slot = {it["sid"]: it["slot"] for it in schedule}
#         # >>> 新增：读取 meta.json 并回填 total_n / chunk_size（若不存在则以 schedule 长度为准）
#         meta = _load_meta(st.session_state.run_dir)
#         if meta:
#             st.session_state.total_n = int(meta.get("total_n", len(schedule)))
#             st.session_state.chunk_size = int(meta.get("chunk_size", st.session_state.chunk_size))
#         else:
#             st.session_state.total_n = len(schedule)
#
#         # （可选但推荐）回填 chunks_total 供 UI 使用
#         st.session_state.chunks_total = math.ceil(st.session_state.total_n / st.session_state.chunk_size)
#
#         # 扫描已落盘记录，恢复 completed/pending/next_sid
#         done = _scan_completed_ids(st.session_state.run_dir)
#         st.session_state.completed_sids = set(done)
#
#         # 用最小堆装载尚未完成的 sid
#         st.session_state.pending = [sid for sid in range(1, st.session_state.total_n + 1) if sid not in done]
#         heapq.heapify(st.session_state.pending)
#
#         # next_sid = 最小未完成
#         ns = 1
#         while ns in st.session_state.completed_sids and ns <= st.session_state.total_n:
#             ns += 1
#         st.session_state.next_sid = ns
#     _st_rerun()


# ----------- 进度条容器 -----------
chunk_prog_box = st.empty()
prog = st.empty()
status = st.empty()
preview_live = st.container()
console = st.container()
cards = st.container()

def _render_qpm_monitor():
    pool = st.session_state.get("provider_pool")
    if not pool:
        return

    # ProviderPool 内部可能是 pool.clients 或 pool.providers，看你的 providers.py 实现
    clients = getattr(pool, "clients", None) or getattr(pool, "providers", None)
    if not clients:
        return

    rows = []
    for idx, c in enumerate(clients, start=1):
        stats = None

        # 优先：如果封装里有 get_qpm_stats()
        if hasattr(c, "get_qpm_stats"):
            try:
                stats = c.get_qpm_stats()
            except Exception:
                stats = None

        # 退化：直接从 limiter 拿 snapshot
        elif hasattr(c, "limiter") and hasattr(c.limiter, "get_snapshot"):
            snap = c.limiter.get_snapshot()
            label = getattr(c, "label", f"client-{idx}")
            stats = {"label": label, **snap}

        if stats:
            rows.append(stats)

    if not rows:
        return

    with st.expander("🧮 Key QPM 实时监控", expanded=True):
        st.dataframe(
            rows,
            use_container_width=True,
            height=min(400, 40 + 24 * len(rows)),
        )

# ----------- 主循环 -----------
if st.session_state.running:
    total_n = st.session_state.total_n
    chunk_size = st.session_state.chunk_size
    chunks_total = math.ceil(total_n / chunk_size)

    # 当前应写 chunk 与片内进度
    next_sid = st.session_state.next_sid
    if next_sid > total_n:
        cur_chunk_idx = chunks_total
        in_chunk_idx = chunk_size if total_n % chunk_size == 0 else total_n % chunk_size
    else:
        cur_chunk_idx = (next_sid - 1) // chunk_size + 1
        chunk_start = (cur_chunk_idx - 1) * chunk_size + 1
        chunk_end = min(cur_chunk_idx * chunk_size, total_n)
        current_chunk_total = chunk_end - chunk_start + 1
        in_chunk_idx = max(0, min(current_chunk_total, next_sid - chunk_start))

    # 顶部进度
    chunk_prog = (cur_chunk_idx - 1) / max(1, chunks_total)
    chunk_prog_box.progress(
        min(chunk_prog, 1.0),
        text=f"分片进度：应写第 {cur_chunk_idx}/{chunks_total} 片（每片 {chunk_size} 条） · 输出目录：{st.session_state.run_dir or '（未初始化）'}"
    )
    if st.session_state.paused:
        status.warning(f"已暂停（累计完成 {len(st.session_state.completed_sids)}/{total_n} 条；继续后自动填补最小缺口 sid={next_sid}）")
    else:
        status.info(f"生成中：已完成 {len(st.session_state.completed_sids)}/{total_n} 条 · 正在填补最小缺口 sid={next_sid}")

    # 片内进度条
    current_chunk_total = min(chunk_size, total_n - (cur_chunk_idx - 1) * chunk_size)
    prog.progress(in_chunk_idx / max(1, current_chunk_total), text=f"当前片进度：{in_chunk_idx}/{current_chunk_total}")

    # 片尾预览：看“应写片”
    with cards:
        st.subheader("📂 当前片末尾预览（来自本地文件）")
        if st.session_state.run_dir:
            preview_items = _load_chunk_preview(st.session_state.run_dir, cur_chunk_idx, max_items=6)
            if preview_items:
                # 仅展示文件尾部若干条
                for idx, item in enumerate(preview_items, start=1):
                    with st.expander(f"#{item.get('id')} — {item.get('姓名')}（{item.get('年级')}） · 代理名：{item.get('代理名')}", expanded=False):
                        st.json(item, expanded=False)
            else:
                st.info("当前片暂无已写入记录。")

    _render_qpm_monitor()


    # 推动一个槽位（未暂停时）
    if not st.session_state.paused:
        # === 结束条件 ===
        if len(st.session_state.completed_sids) >= st.session_state.total_n or len(st.session_state.pending) == 0:
            prog.progress(1.0, text="当前片进度：完成 ✅")
            chunk_prog_box.progress(1.0, text="分片进度：全部完成 ✅")
            status.success(f"全部生成完成！文件已保存到：{st.session_state.run_dir}")
            st.session_state.running = False
            _st_rerun()

        # === 准备本步的 sid 批次（最小缺口优先） ===
        C_star = _estimate_concurrency(st.session_state.provider_pool, kappa=1.3, beta=0.7, cmax=1024)
        _log_debug(f"开始批处理：准备生成 {len(st.session_state.pending)} 个待处理条目，预估并发C*={C_star}")

        batch_size = max(int(per_step_m), int(2 * C_star))
        scheduled_sids = []
        for _ in range(min(batch_size, len(st.session_state.pending))):
            scheduled_sids.append(heapq.heappop(st.session_state.pending))
        if not scheduled_sids:
            _st_rerun()

        _log_debug(f"本批次处理 {len(scheduled_sids)} 个条目: {scheduled_sids}")


        # —— 使用 ProviderPool 动态挑选最佳提供者 ——
        pool = st.session_state.provider_pool
        if pool is None:
            st.error("提供者池未就绪：请先在侧边栏设置 secrets/key.txt 并重载。")
            st.stop()


        def _pick_client() -> ProviderClient:
            # 内含 AIMD 限流 + UCB 探索 + 容量惩罚的评分；每次 pick 可能得到不同厂商
            return pool.pick()


        accepted = 0
        last_error = None
        max_workers = max(1, min(len(scheduled_sids), int(C_star)))
        _log_debug(f"使用线程池：max_workers={max_workers}")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            _log_debug(f"开始提交 {len(scheduled_sids)} 个任务到线程池")
            for sid in scheduled_sids:
                client = _pick_client()
                slot = st.session_state.sid2slot.get(sid, {})
                _log_debug(f"提交任务 sid={sid}，使用 {client.spec.name}", sid=sid)
                fut = ex.submit(
                    _worker_process_one,
                    st.session_state.orch, sid, slot,
                    client,
                    st.session_state.sim_gate,
                    MAX_RETRIES_PER_SLOT
                )
                futures[fut] = sid
            _log_debug(f"所有任务已提交，等待结果...")

            for fut in as_completed(futures):
                sid = futures[fut]
                ok, item, dialog, err = fut.result()
                if ok and item:
                    _log_debug(f"成功生成 sid={sid}", sid=sid)
                    wrote = _append_record_by_sid(st.session_state.run_dir,
                                                  st.session_state.chunk_size,
                                                  item,
                                                  st.session_state.write_lock)
                    if wrote:
                        accepted += 1
                        _log_debug(f"成功写入 sid={sid}", sid=sid)
                    st.session_state.last_item = item
                    st.session_state.last_dialog = dialog or st.session_state.last_dialog
                else:
                    _log_debug(f"生成失败 sid={sid}: {err}", sid=sid)
                    heapq.heappush(st.session_state.pending, sid)
                    last_error = err or "未知错误"

            st.session_state.last_error = None if accepted == len(scheduled_sids) \
                else f"本步并行完成 {accepted}/{len(scheduled_sids)}，失败已回队重试。"

        # UI 轻提示
        _log_debug(f"批次完成：{accepted}/{len(scheduled_sids)} 成功")
        st.session_state.last_error = None if accepted == len(
            scheduled_sids) else f"本步并行完成 {accepted}/{len(scheduled_sids)}，失败已回队重试。"
        _st_rerun()

