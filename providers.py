# providers.py  —— 稳健加载 + 头部修正
# -*- coding: utf-8 -*-
import os, json, time, math, threading, requests
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# -------- 工具：清洗一行（去 BOM / 零宽空格 / 全角空格） --------
def _clean(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("\ufeff", "")          # BOM
    s = s.replace("\u200b", "")          # zero width space
    s = s.replace("\u00a0", " ")         # NBSP -> 空格
    s = s.strip()
    return s

def _read_lines(fp: str) -> List[str]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"keys file not found: {fp}")
    out = []
    with open(fp, "r", encoding="utf-8") as f:
        for raw in f:
            ln = _clean(raw)
            if ln and not ln.startswith("#"):
                out.append(ln)
    return out

def _parse_line(ln: str) -> Dict[str, Any]:
    # 先尝试 JSON
    try:
        if ln.startswith("{") and ln.endswith("}"):
            obj = json.loads(ln)
            obj.setdefault("qpm", 6)
            obj.setdefault("capacity_max", max(180, int(obj.get("qpm", 6))))
            obj.setdefault("headers", "Authorization:Bearer")
            return obj
    except Exception as e:
        raise ValueError(f"JSON line parse error: {e} :: {ln}")

    # 再尝试管道分隔：name|base_url|api_key|model|qpm=10|headers=api-key
    parts = [p.strip() for p in ln.split("|")]
    if len(parts) < 4:
        raise ValueError(f"bad provider line (expect JSON or pipe-4 fields): {ln}")
    name, base_url, api_key, model = parts[:4]
    kv = {"name": name, "base_url": base_url, "api_key": api_key, "model": model,
          "qpm": 6, "capacity_max": 180, "headers": "Authorization:Bearer"}
    for p in parts[4:]:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip(); v = v.strip()
            kv[k] = int(v) if v.isdigit() else v
        elif ":" in p:  # headers=Authorization:Bearer（也允许直接给 "Authorization:Bearer"）
            kv["headers"] = p.strip()
    return kv

@dataclass
class ProviderSpec:
    name: str
    base_url: str
    api_key: str
    model: str
    qpm: int = 6
    capacity_max: int = 180
    headers: str = "Authorization:Bearer"
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProviderSpec":
        header_spec = _clean(d.get("headers", "Authorization:Bearer"))
        # headers 形如 "Authorization:Bearer" / "api-key" / "x-api-key"
        if ":" in header_spec:
            hdr_key, hdr_val = header_spec.split(":", 1)
            hdr_key, hdr_val = _clean(hdr_key), _clean(hdr_val)
        else:
            hdr_key, hdr_val = _clean(header_spec), "Bearer"
        extra = dict(d.get("extra_headers") or {})
        # 规范化：Authorization: Bearer <key>；api-key: <key>
        if hdr_key.lower() == "authorization":
            scheme = hdr_val if hdr_val else "Bearer"
            extra["Authorization"] = f"{scheme} {d['api_key']}"
        elif hdr_key.lower() in ("api-key", "x-api-key"):
            extra[hdr_key] = d["api_key"]
        else:
            # 兜底：仍使用 Authorization: Bearer
            extra["Authorization"] = f"Bearer {d['api_key']}"
        return ProviderSpec(
            name=d["name"],
            base_url=_clean(d["base_url"]).rstrip("/"),
            api_key=d["api_key"],
            model=d["model"],
            qpm=int(d.get("qpm", 6)),
            capacity_max=int(d.get("capacity_max", max(180, int(d.get("qpm", 6))))),
            headers=header_spec,
            extra_headers=extra
        )

# -------- 令牌桶 + AIMD --------
class RateLimiter:
    def __init__(self, qpm_init: int = 6, capacity_max: int = 180):
        self.capacity = max(1, int(qpm_init))
        self.capacity_max = max(self.capacity, int(capacity_max))
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()
        self.window = 60.0
        self.lock = threading.Lock()
        self.success_streak = 0
        self.additive_every = 20
        self.additive_step = 1
        self.beta = 0.7
        self.violation_count = 0  # 新增：累计 429/503 次数

    def _refill(self):
        now = time.monotonic()
        el = now - self.updated
        if el <= 0: return
        refill = el * (self.capacity / self.window)
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

    def on_violation(self):
        with self.lock:
            self.capacity = max(1, int(self.capacity * self.beta))
            self.tokens = min(self.tokens, self.capacity)
            self.success_streak = 0
            self.violation_count += 1  # 新增

    def get_snapshot(self) -> Dict[str, float]:
        with self.lock:
            return {
                "capacity_qpm": float(self.capacity),  # 要显示的“QPM”
                "tokens": float(self.tokens),
                "violations": int(self.violation_count),
            }


# -------- 统一客户端 --------
class ProviderClient:
    def __init__(self, spec: ProviderSpec):
        self.spec = spec
        self.headers = {"Content-Type": "application/json", **spec.extra_headers}
        self.limiter = RateLimiter(qpm_init=spec.qpm, capacity_max=spec.capacity_max)
        # 统计量（EWMA）
        self.lock = threading.Lock()
        self.n = 0
        self.lat_ewma = None
        self.succ_ewma = None
        tail = self.spec.api_key[-6:] if len(self.spec.api_key) >= 6 else self.spec.api_key
        self.label = f"{self.spec.base_url}…{tail}"  # 新增：前端显示名

    def _update_stats(self, ok: bool, latency: float):
        with self.lock:
            self.n += 1
            a = 0.2
            self.lat_ewma = latency if self.lat_ewma is None else (1-a)*self.lat_ewma + a*latency
            s = 1.0 if ok else 0.0
            self.succ_ewma = s if self.succ_ewma is None else (1-a)*self.succ_ewma + a*s

    def post(self, path: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        payload = dict(payload)
        payload.setdefault("model", self.spec.model)
        url = f"{self.spec.base_url}{path if path.startswith('/') else '/' + path}"
        self.limiter.acquire()
        backoff = 0.8
        t0 = time.monotonic()
        for _ in range(4):
            r = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
            if r.status_code in (429, 503):
                ra = r.headers.get("Retry-After")
                try:
                    if ra: time.sleep(float(ra))
                except Exception:
                    pass
                time.sleep(backoff); backoff *= 1.8
                self.limiter.on_violation()
                continue
            ok = (200 <= r.status_code < 300)
            self._update_stats(ok, time.monotonic() - t0)
            r.raise_for_status()
            self.limiter.on_success()
            return r.json()
        self._update_stats(False, time.monotonic() - t0)
        r.raise_for_status()

    def score(self, t_global: int, alpha: float=0.8, gamma: float=0.5, rho: float=0.8) -> float:
        tau = self.lat_ewma if self.lat_ewma is not None else 1.0
        p = self.succ_ewma if self.succ_ewma is not None else 0.9
        with self.limiter.lock:
            cap = max(1.0, float(self.limiter.capacity))
            used_ratio = 1.0 - min(1.0, self.limiter.tokens / cap)
        cap_penalty = max(0.0, used_ratio - rho)
        n = max(1, self.n)
        ucb = math.sqrt(max(1.0, math.log(max(2, t_global))) / n)
        return tau + alpha*(1.0 - p) + gamma*cap_penalty - ucb

    def get_qpm_stats(self) -> Dict[str, Any]:
        snap = self.limiter.get_snapshot()
        snap["label"] = getattr(self, "label", self.spec.base_url)
        snap["model"] = self.spec.model  # 可选：方便排查
        return snap

class ProviderPool:
    def __init__(self, clients: List[ProviderClient]):
        if not clients:
            raise ValueError("no providers loaded")
        self.clients = clients
        self._t = 1

    def pick(self) -> ProviderClient:
        best = min(self.clients, key=lambda c: c.score(self._t))
        self._t += 1
        return best

def load_providers(keys_path: str) -> List[ProviderClient]:
    # 允许传相对路径；转绝对路径（以当前脚本文件为基准更稳妥）
    if not os.path.isabs(keys_path):
        base = os.path.dirname(os.path.abspath(__file__))
        keys_path = os.path.join(base, keys_path)

    if not os.path.exists(keys_path):
        raise FileNotFoundError(f"keys file not found: {keys_path}")

    clients: List[ProviderClient] = []
    errors: List[str] = []

    # 尝试解析为完整的 JSON 数组格式
    try:
        with open(keys_path, "r", encoding="utf-8") as f:
            # 尝试将整个文件解析为 JSON
            first_char = f.read(1)
            if not first_char:
                raise ValueError("provider config file is empty")
            f.seek(0)  # 重置文件指针

            # 如果是 JSON 数组格式
            if first_char == "[":
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        # 成功解析为 JSON 数组
                        for idx, obj in enumerate(data, start=1):
                            if not isinstance(obj, dict):
                                errors.append(f"line {idx}: not a JSON object")
                                continue
                            try:
                                spec = ProviderSpec.from_dict(obj)
                                clients.append(ProviderClient(spec))
                            except Exception as e:
                                errors.append(f"line {idx}: {e}")
                        # 如果成功加载至少一个，就返回（忽略后续处理）
                        if clients:
                            return clients
                        # 如果没有成功加载任何 provider，继续尝试其他格式
                except json.JSONDecodeError as e:
                    # JSON 解析失败，回退到逐行解析
                    pass

    except Exception as e:
        errors.append(f"JSON array parse error: {e}")

    # 回退到原有的逐行解析逻辑（处理 JSONL 或管道分隔格式）
    lines = _read_lines(keys_path)
    for idx, ln in enumerate(lines, start=1):
        try:
            spec = ProviderSpec.from_dict(_parse_line(ln))
            clients.append(ProviderClient(spec))
        except Exception as e:
            errors.append(f"line {idx}: {e}")

    if not clients:
        detail = "; ".join(errors) if errors else "empty file?"
        raise ValueError(f"no providers loaded — details: {detail}")
    return clients

# 线程局部（可与原逻辑配合）
_TLS = threading.local()
def set_client_for_thread(client: ProviderClient): setattr(_TLS, "client", client)
def get_client_for_thread() -> Optional[ProviderClient]: return getattr(_TLS, "client", None)
