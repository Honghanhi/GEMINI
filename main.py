"""
AI-PROOF Backend  —  FastAPI + Gemini API
Deploy: Render.com (Free tier)

ENV VARS:
  GEMINI_API_KEY=your_key_here           ← bắt buộc
  GEMINI_API_KEYS=key1,key2,key3        ← tuỳ chọn, multi-key load balancing
  VIRUSTOTAL_API_KEY=your_key_here      ← tuỳ chọn
  ALLOWED_ORIGINS=https://your-site.com ← tuỳ chọn, mặc định *
  CACHE_TTL=300                          ← giây, mặc định 300
"""

import os, json, asyncio, time, random, hashlib, logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aiproof")

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
# Multi-key: ưu tiên GEMINI_API_KEYS (comma-separated), fallback GEMINI_API_KEY
_raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]

VT_KEY    = os.getenv("VIRUSTOTAL_API_KEY", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# Models theo thứ tự ưu tiên — chỉ dùng endpoint v1beta stable
GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

TIMEOUT = httpx.Timeout(25.0, connect=8.0)

app = FastAPI(title="AI-PROOF Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
#  CACHE  (in-memory, đủ dùng 1 worker)
# ══════════════════════════════════════════════════════════
_cache: dict[str, tuple[float, dict]] = {}  # key → (expire_ts, value)

def _cache_get(key: str) -> dict | None:
    entry = _cache.get(key)
    if entry and entry[0] > time.time():
        return entry[1]
    _cache.pop(key, None)
    return None

def _cache_set(key: str, value: dict, ttl: int = CACHE_TTL):
    # Giữ cache không quá 500 entries (LRU đơn giản)
    if len(_cache) >= 500:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)
    _cache[key] = (time.time() + ttl, value)

def _cache_key(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

# ══════════════════════════════════════════════════════════
#  GEMINI KEY ROTATION
# ══════════════════════════════════════════════════════════
class KeyPool:
    """Round-robin key pool với per-key cooldown."""
    def __init__(self, keys: list[str]):
        self._keys = keys
        self._cooldown: dict[str, float] = {}  # key → resume_ts
        self._idx = 0

    def get(self) -> str | None:
        now = time.time()
        n = len(self._keys)
        for _ in range(n):
            key = self._keys[self._idx % n]
            self._idx += 1
            if self._cooldown.get(key, 0) <= now:
                return key
        return None  # tất cả đều cooldown

    def penalize(self, key: str, seconds: int):
        self._cooldown[key] = time.time() + seconds
        log.warning(f"Key ...{key[-6:]} penalized {seconds}s")

key_pool = KeyPool(GEMINI_KEYS)

# ══════════════════════════════════════════════════════════
#  MODEL HEALTH  (tự loại model 404 tạm thời)
# ══════════════════════════════════════════════════════════
_model_dead_until: dict[str, float] = {}  # model → resume_ts

def _live_models() -> list[str]:
    now = time.time()
    return [m for m in GEMINI_MODELS if _model_dead_until.get(m, 0) <= now]

def _kill_model(model: str, seconds: int = 300):
    _model_dead_until[model] = time.time() + seconds
    log.warning(f"Model {model} disabled {seconds}s")

# ══════════════════════════════════════════════════════════
#  GEMINI CALL — CORE
# ══════════════════════════════════════════════════════════
_gemini_sem = asyncio.Semaphore(3)  # tối đa 3 concurrent Gemini calls

async def _gemini_call(prompt: str, max_tokens: int = 1000) -> str:
    """
    Gọi Gemini với:
    - Multi-key round-robin
    - Per-model fallback (bỏ qua model 404)
    - Exponential backoff + jitter cho 429
    - Giới hạn retry rõ ràng (không loop vô tận)
    - Semaphore chống spam request
    """
    if not GEMINI_KEYS:
        raise HTTPException(503, detail="Chưa cấu hình GEMINI_API_KEY")

    models = _live_models()
    if not models:
        raise HTTPException(503, detail="Tất cả Gemini models tạm thời không khả dụng")

    async with _gemini_sem:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for model in models:
                # Thử mỗi model tối đa 2 lần (1 lần 429 retry)
                for attempt in range(2):
                    key = key_pool.get()
                    if not key:
                        # Tất cả keys đang cooldown — chờ ngắn rồi báo lỗi
                        wait = 10
                        log.warning(f"All keys in cooldown, waiting {wait}s")
                        await asyncio.sleep(wait)
                        key = key_pool.get()
                        if not key:
                            raise HTTPException(429, detail="Quota Gemini tạm thời hết, thử lại sau ít phút")

                    url = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{model}:generateContent?key={key}"
                    )
                    body = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "maxOutputTokens": max_tokens,
                            "temperature": 0.2,
                        },
                    }

                    try:
                        res = await client.post(url, json=body)
                    except httpx.TimeoutException:
                        log.warning(f"Timeout {model} attempt {attempt+1}")
                        break  # thử model tiếp theo
                    except httpx.RequestError as e:
                        log.warning(f"Network error {model}: {e}")
                        break

                    # ── Thành công ──────────────────────────────────
                    if res.status_code == 200:
                        try:
                            data = res.json()
                            text = (
                                data["candidates"][0]["content"]["parts"][0]["text"]
                            )
                            return text
                        except (KeyError, IndexError, json.JSONDecodeError) as e:
                            log.error(f"Parse error {model}: {e} | raw: {res.text[:200]}")
                            break  # thử model khác

                    # ── Rate limit ──────────────────────────────────
                    if res.status_code == 429:
                        retry_after = int(res.headers.get("Retry-After", "0"))
                        # Backoff: max(retry_after, base) + jitter
                        base = 15 * (2 ** attempt)        # 15s, 30s
                        wait = max(retry_after, base) + random.uniform(0, 5)
                        wait = min(wait, 60)               # không chờ quá 60s
                        log.warning(f"429 {model} key ...{key[-6:]}, backoff {wait:.1f}s")
                        key_pool.penalize(key, int(wait))
                        if attempt == 0:
                            await asyncio.sleep(wait)
                            continue  # retry cùng model, key khác
                        break  # attempt 1 vẫn 429 → thử model tiếp

                    # ── Model không tồn tại ─────────────────────────
                    if res.status_code == 404:
                        _kill_model(model, 600)  # disable 10 phút
                        break  # thử model tiếp ngay

                    # ── Lỗi xác thực / quota cứng ──────────────────
                    if res.status_code in (400, 403):
                        detail = res.text[:300]
                        log.error(f"Hard error {res.status_code} {model}: {detail}")
                        raise HTTPException(res.status_code, detail=f"Gemini: {detail}")

                    # ── Lỗi khác (5xx) ─────────────────────────────
                    log.warning(f"Unexpected {res.status_code} {model}")
                    break

    raise HTTPException(503, detail="Gemini không phản hồi được. Thử lại sau.")


# ══════════════════════════════════════════════════════════
#  JSON PARSE HELPER
# ══════════════════════════════════════════════════════════
def _parse_json(text: str) -> dict:
    """Parse JSON từ Gemini output, bỏ markdown fences an toàn."""
    text = text.strip()

    # Bỏ ```json ... ``` hoặc ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        # Bỏ dòng đầu (```json) và dòng cuối (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    # Tìm object JSON đầu tiên
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed: {e} | text: {text[:300]}")
        raise HTTPException(502, detail=f"Gemini trả về JSON không hợp lệ: {str(e)}")


# ══════════════════════════════════════════════════════════
#  REQUEST MODELS
# ══════════════════════════════════════════════════════════
class AnalyzeURLRequest(BaseModel):
    url: str
    threatScore:  Optional[int]  = 0
    threats:      Optional[list] = []
    hasGambling:  Optional[bool] = False
    vtMalicious:  Optional[int]  = 0
    vtSuspicious: Optional[int]  = 0
    country:      Optional[str]  = None
    org:          Optional[str]  = None

class AnalyzeTextRequest(BaseModel):
    text:  str
    title: Optional[str] = ""

class ProxyAIRequest(BaseModel):
    messages:   list
    max_tokens: Optional[int] = 1000


# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "gemini_keys": len(GEMINI_KEYS),
        "virustotal":  bool(VT_KEY),
        "live_models": _live_models(),
        "cache_size":  len(_cache),
    }


# ── /analyze/url ──────────────────────────────────────────
@app.post("/analyze/url")
async def analyze_url(req: AnalyzeURLRequest):
    ck = _cache_key("url", req.url, req.threatScore, req.vtMalicious, req.hasGambling)
    if cached := _cache_get(ck):
        return {**cached, "_cached": True}

    prompt = f"""Bạn là chuyên gia an ninh mạng Việt Nam. Phân tích URL sau dựa trên dữ liệu được cung cấp.

URL: {req.url}
Threat Score hiện tại: {req.threatScore}/100
VirusTotal: {req.vtMalicious} độc hại, {req.vtSuspicious} nghi ngờ
Gambling phát hiện: {"CÓ" if req.hasGambling else "Không"}
Quốc gia server: {req.country or "Không rõ"}
ISP/Org: {req.org or "Không rõ"}
Các mối đe dọa: {", ".join(req.threats) if req.threats else "Không có"}

Trả về JSON hợp lệ (KHÔNG markdown, KHÔNG text ngoài JSON):
{{
  "verdict": "safe|caution|dangerous|phishing|malware|gambling",
  "confidence": 0-100,
  "safetyScore": 0-100,
  "websiteType": "loại website ngắn gọn",
  "websiteDescription": "mô tả 1 câu tiếng Việt",
  "narrative": "nhận xét 2-3 câu tiếng Việt dựa trên dữ liệu",
  "warnings": ["cảnh báo cụ thể nếu có"],
  "trustFactors": ["điểm tích cực nếu có"],
  "recommendations": ["khuyến nghị cho người dùng VN"],
  "isVietnamese": true,
  "vietnameseNotes": "ghi chú nếu là web VN hoặc nhắm vào người VN"
}}"""

    raw  = await _gemini_call(prompt, 900)
    data = _parse_json(raw)
    result = {"ok": True, **data}
    _cache_set(ck, result)
    return result


# ── /analyze/text ─────────────────────────────────────────
@app.post("/analyze/text")
async def analyze_text(req: AnalyzeTextRequest):
    combined = f"{req.title}\n\n{req.text}".strip()[:3000]
    ck = _cache_key("text", combined)
    if cached := _cache_get(ck):
        return {**cached, "_cached": True}

    prompt = f"""Bạn là fact-checker chuyên nghiệp. Phân tích nội dung sau:

{combined}

Trả về JSON hợp lệ (KHÔNG markdown, KHÔNG text ngoài JSON):
{{
  "trustScore": 0-100,
  "verdict": "authentic|misleading|satire|fabricated|uncertain",
  "fakePct": 0-100,
  "realPct": 0-100,
  "confidence": 0-100,
  "summary": "tóm tắt đánh giá 1-2 câu tiếng Việt",
  "biasType": "political|emotional|sensational|commercial|none",
  "redFlags": ["dấu hiệu đáng ngờ"],
  "trustFactors": ["yếu tố tích cực"],
  "overallAssessment": "đánh giá tổng thể 2-3 câu tiếng Việt"
}}"""

    raw  = await _gemini_call(prompt, 800)
    data = _parse_json(raw)
    result = {"ok": True, **data}
    _cache_set(ck, result)
    return result


# ── /proxy/ai ─────────────────────────────────────────────
@app.post("/proxy/ai")
async def proxy_ai(req: ProxyAIRequest):
    # Lấy message user cuối cùng
    prompt = next(
        (m.get("content", "") for m in reversed(req.messages) if m.get("role") == "user"),
        ""
    )
    if not prompt:
        raise HTTPException(400, detail="Không có nội dung user message")

    text = await _gemini_call(prompt, min(req.max_tokens, 1500))
    return {"ok": True, "text": text}


# ── /proxy/dns ────────────────────────────────────────────
@app.post("/proxy/dns")
async def proxy_dns(body: dict):
    domain = body.get("domain", "").strip()
    qtype  = body.get("type", "A")
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    ck = _cache_key("dns", domain, qtype)
    if cached := _cache_get(ck):
        return cached

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            res = await client.get(
                "https://dns.google/resolve",
                params={"name": domain, "type": qtype},
                headers={"Accept": "application/dns-json"},
            )
            res.raise_for_status()
            data    = res.json()
            records = [a["data"] for a in (data.get("Answer") or []) if a.get("type") == 1]
            result  = {
                "ok":      data.get("Status") == 0,
                "records": records,
                "answers": data.get("Answer") or [],
                "domain":  domain,
                "type":    qtype,
            }
        except Exception as e:
            log.warning(f"DNS error {domain}: {e}")
            result = {"ok": False, "records": [], "domain": domain, "type": qtype}

    _cache_set(ck, result, ttl=60)
    return result


# ── /proxy/ipinfo ─────────────────────────────────────────
CF_PREFIXES = (
    "172.64.","172.65.","172.66.","172.67.","172.68.","172.69.",
    "172.70.","172.71.","172.72.","172.73.","172.74.","172.75.",
    "172.76.","172.77.","172.78.","172.79.","172.80.","172.81.",
    "172.82.","172.83.","172.84.","172.85.","172.86.","172.87.",
    "172.88.","172.89.","172.90.","172.91.","172.92.","172.93.",
    "172.94.","172.95.","172.96.","172.97.","172.98.","172.99.",
    "172.100.","172.101.","172.102.","172.103.","172.104.","172.105.",
    "172.106.","172.107.","172.108.","172.109.","172.110.","172.111.",
    "172.112.","172.113.","172.114.","172.115.","172.116.","172.117.",
    "172.118.","172.119.","172.120.","172.121.","172.122.","172.123.",
    "172.124.","172.125.","172.126.","172.127.","172.128.","172.129.",
    "172.130.","172.131.",
    "104.16.","104.17.","104.18.","104.19.","104.20.","104.21.",
    "104.22.","104.23.","104.24.","104.25.","104.26.","104.27.",
    "104.28.","104.29.","104.30.","104.31.",
    "162.158.","141.101.64.","141.101.65.","141.101.66.","141.101.67.",
    "188.114.96.","188.114.97.","190.93.240.","198.41.128.",
)

def _is_cloudflare(ip: str) -> bool:
    return any(ip.startswith(p) for p in CF_PREFIXES)

@app.post("/proxy/ipinfo")
async def proxy_ipinfo(body: dict):
    domain = body.get("domain", "").strip()
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    ck = _cache_key("ipinfo", domain)
    if cached := _cache_get(ck):
        return cached

    ip = domain
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Resolve IP
        try:
            dns_res = await client.get(
                "https://dns.google/resolve",
                params={"name": domain, "type": "A"},
                headers={"Accept": "application/dns-json"},
            )
            answers  = dns_res.json().get("Answer") or []
            a_recs   = [a["data"] for a in answers if a.get("type") == 1]
            if a_recs:
                ip = a_recs[0]
        except Exception:
            pass

        if _is_cloudflare(ip):
            result = {
                "ok": True, "ip": ip,
                "country": "Cloudflare CDN", "countryCode": "CF",
                "region": "", "city": "",
                "org": "Cloudflare, Inc.", "isp": "Cloudflare",
                "asn": "AS13335", "isVPN": False, "isTor": False, "isHosting": True,
                "_note": "Cloudflare CDN — không phản ánh server thật",
            }
            _cache_set(ck, result, ttl=3600)
            return result

        # ip-api.com
        try:
            res = await client.get(
                f"https://ip-api.com/json/{ip}",
                params={"fields": "status,country,countryCode,regionName,city,org,isp,as,proxy,hosting,query"},
            )
            if res.is_success:
                d = res.json()
                if d.get("status") == "success":
                    result = {
                        "ok": True, "ip": d.get("query", ip),
                        "country":     d.get("country"),
                        "countryCode": d.get("countryCode"),
                        "region":      d.get("regionName"),
                        "city":        d.get("city"),
                        "org":         d.get("org"),
                        "isp":         d.get("isp"),
                        "asn":         d.get("as"),
                        "isVPN":       d.get("proxy", False),
                        "isTor":       False,
                        "isHosting":   d.get("hosting", False),
                    }
                    _cache_set(ck, result, ttl=3600)
                    return result
        except Exception as e:
            log.warning(f"ip-api error {ip}: {e}")

    result = {"ok": False, "ip": ip}
    return result


# ── /proxy/virustotal/url ─────────────────────────────────
@app.post("/proxy/virustotal/url")
async def proxy_vt_url(body: dict):
    if not VT_KEY:
        return {"ok": False, "noKey": True}
    url = body.get("url", "")
    if not url:
        raise HTTPException(400, detail="Thiếu url")

    ck = _cache_key("vt_url", url)
    if cached := _cache_get(ck):
        return cached

    import base64
    encoded = base64.urlsafe_b64encode(url.encode()).rstrip(b"=").decode()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(
            f"https://www.virustotal.com/api/v3/urls/{encoded}",
            headers={"x-apikey": VT_KEY},
        )
        if not res.is_success:
            sub = await client.post(
                "https://www.virustotal.com/api/v3/urls",
                headers={"x-apikey": VT_KEY, "Content-Type": "application/x-www-form-urlencoded"},
                content=f"url={url}",
            )
            if not sub.is_success:
                return {"ok": False}
            return {"ok": True, "malicious": 0, "suspicious": 0, "totalEngines": 0, "submitted": True}

    stats = res.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    result = {
        "ok":           True,
        "malicious":    stats.get("malicious", 0),
        "suspicious":   stats.get("suspicious", 0),
        "harmless":     stats.get("harmless", 0),
        "undetected":   stats.get("undetected", 0),
        "totalEngines": sum(stats.values()),
    }
    _cache_set(ck, result, ttl=600)
    return result


# ── /proxy/virustotal/domain ──────────────────────────────
@app.post("/proxy/virustotal/domain")
async def proxy_vt_domain(body: dict):
    if not VT_KEY:
        return {"ok": False, "noKey": True}
    domain = body.get("domain", "")
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    ck = _cache_key("vt_domain", domain)
    if cached := _cache_get(ck):
        return cached

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(
            f"https://www.virustotal.com/api/v3/domains/{domain}",
            headers={"x-apikey": VT_KEY},
        )
    if not res.is_success:
        return {"ok": False}

    import datetime
    attr = res.json().get("data", {}).get("attributes", {})
    cd   = attr.get("creation_date")
    result = {
        "ok":           True,
        "reputation":   attr.get("reputation", 0),
        "categories":   attr.get("categories", {}),
        "country":      attr.get("country"),
        "creationDate": datetime.datetime.utcfromtimestamp(cd).strftime("%Y-%m-%d") if cd else None,
    }
    _cache_set(ck, result, ttl=3600)
    return result


# ── /proxy/urlscan ────────────────────────────────────────
@app.post("/proxy/urlscan")
async def proxy_urlscan(body: dict):
    domain = body.get("domain", "")
    ck = _cache_key("urlscan", domain)
    if cached := _cache_get(ck):
        return cached

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            res = await client.get(
                "https://urlscan.io/api/v1/search/",
                params={"q": f"domain:{domain}", "size": "3", "sort": "date"},
            )
            if not res.is_success:
                return {"ok": False, "malicious": False, "tags": []}
            data    = res.json()
            results = data.get("results", [])
            is_mal  = any(
                r.get("verdicts", {}).get("overall", {}).get("malicious")
                for r in results
            )
            tags   = list({
                t for r in results
                for t in r.get("verdicts", {}).get("overall", {}).get("tags", [])
            })
            server = results[0].get("page", {}).get("server") if results else None
            result = {"ok": True, "malicious": is_mal, "tags": tags, "server": server}
            _cache_set(ck, result, ttl=300)
            return result
        except Exception as e:
            log.warning(f"urlscan error {domain}: {e}")
            return {"ok": False, "malicious": False, "tags": []}


# ── /proxy/allorigins ─────────────────────────────────────
@app.post("/proxy/allorigins")
async def proxy_allorigins(body: dict):
    url = body.get("url", "")
    if not url:
        raise HTTPException(400, detail="Thiếu url")
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        try:
            res = await client.get("https://api.allorigins.win/raw", params={"url": url})
            if res.is_success:
                return {"ok": True, "html": res.text[:300_000]}
        except Exception as e:
            log.warning(f"allorigins error {url}: {e}")
    return {"ok": False, "html": ""}


# ══════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        workers=1,   # 1 worker → semaphore hoạt động đúng
    )
