"""
AI-PROOF Backend  —  FastAPI + Gemini API
Python 3.9.18 compatible

ENV VARS:
  GEMINI_API_KEY=your_key_here           ← bắt buộc
  GEMINI_API_KEYS=key1,key2,key3         ← tuỳ chọn, multi-key load balancing
  VIRUSTOTAL_API_KEY=your_key_here       ← tuỳ chọn
  ALLOWED_ORIGINS=https://your-site.com  ← tuỳ chọn, mặc định *
  CACHE_TTL=300                          ← giây, mặc định 300
"""

import os, json, asyncio, time, random, hashlib, logging, base64, datetime
from typing import Dict, List, Optional, Tuple, Any

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
_raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_KEYS: List[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]

VT_KEY    = os.getenv("VIRUSTOTAL_API_KEY", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

GEMINI_MODELS: List[str] = [
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

TIMEOUT = httpx.Timeout(25.0, connect=8.0)

app = FastAPI(title="AI-PROOF Backend", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
#  CACHE  (in-memory)
# ══════════════════════════════════════════════════════════
_cache: Dict[str, Tuple[float, Dict]] = {}


def _cache_get(key: str) -> Optional[Dict]:
    entry = _cache.get(key)
    if entry and entry[0] > time.time():
        return entry[1]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: Dict, ttl: int = CACHE_TTL) -> None:
    if len(_cache) >= 500:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)
    _cache[key] = (time.time() + ttl, value)


def _cache_key(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ══════════════════════════════════════════════════════════
#  KEY POOL
# ══════════════════════════════════════════════════════════
class KeyPool:
    """Round-robin key pool với per-key cooldown."""

    def __init__(self, keys: List[str]) -> None:
        self._keys = keys
        self._cooldown: Dict[str, float] = {}
        self._idx = 0

    def get(self) -> Optional[str]:
        now = time.time()
        n = len(self._keys)
        for _ in range(n):
            key = self._keys[self._idx % n]
            self._idx += 1
            if self._cooldown.get(key, 0) <= now:
                return key
        return None

    def penalize(self, key: str, seconds: int) -> None:
        self._cooldown[key] = time.time() + seconds
        log.warning("Key ...%s penalized %ds", key[-6:], seconds)


key_pool = KeyPool(GEMINI_KEYS)

# ══════════════════════════════════════════════════════════
#  MODEL HEALTH
# ══════════════════════════════════════════════════════════
_model_dead_until: Dict[str, float] = {}


def _live_models() -> List[str]:
    now = time.time()
    return [m for m in GEMINI_MODELS if _model_dead_until.get(m, 0) <= now]


def _kill_model(model: str, seconds: int = 300) -> None:
    _model_dead_until[model] = time.time() + seconds
    log.warning("Model %s disabled %ds", model, seconds)


# ══════════════════════════════════════════════════════════
#  GEMINI CALL
# ══════════════════════════════════════════════════════════
_gemini_sem = asyncio.Semaphore(3)


async def _gemini_call(prompt: str, max_tokens: int = 1000) -> str:
    if not GEMINI_KEYS:
        raise HTTPException(503, detail="Chưa cấu hình GEMINI_API_KEY")

    models = _live_models()
    if not models:
        raise HTTPException(503, detail="Tất cả Gemini models tạm thời không khả dụng")

    async with _gemini_sem:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for model in models:
                for attempt in range(2):
                    key = key_pool.get()
                    if not key:
                        await asyncio.sleep(10)
                        key = key_pool.get()
                        if not key:
                            raise HTTPException(429, detail="Quota Gemini tạm thời hết, thử lại sau ít phút")

                    url = (
                        "https://generativelanguage.googleapis.com/v1beta/models/"
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
                        log.warning("Timeout %s attempt %d", model, attempt + 1)
                        break
                    except httpx.RequestError as e:
                        log.warning("Network error %s: %s", model, e)
                        break

                    if res.status_code == 200:
                        try:
                            data = res.json()
                            text = (
                                data["candidates"][0]["content"]["parts"][0]["text"]
                            )
                            return text
                        except (KeyError, IndexError, json.JSONDecodeError) as e:
                            log.error("Parse error %s: %s | raw: %s", model, e, res.text[:200])
                            break

                    if res.status_code == 429:
                        retry_after = int(res.headers.get("Retry-After", "0"))
                        base = 15 * (2 ** attempt)
                        wait = max(retry_after, base) + random.uniform(0, 5)
                        wait = min(wait, 60)
                        log.warning("429 %s key ...%s, backoff %.1fs", model, key[-6:], wait)
                        key_pool.penalize(key, int(wait))
                        if attempt == 0:
                            await asyncio.sleep(wait)
                            continue
                        break

                    if res.status_code == 404:
                        _kill_model(model, 600)
                        break

                    if res.status_code in (400, 403):
                        detail = res.text[:300]
                        log.error("Hard error %d %s: %s", res.status_code, model, detail)
                        raise HTTPException(res.status_code, detail=f"Gemini: {detail}")

                    log.warning("Unexpected %d %s", res.status_code, model)
                    break

    raise HTTPException(503, detail="Gemini không phản hồi được. Thử lại sau.")


# ══════════════════════════════════════════════════════════
#  JSON PARSE
# ══════════════════════════════════════════════════════════
def _parse_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("JSON parse failed: %s | text: %s", e, text[:300])
        raise HTTPException(502, detail=f"Gemini trả về JSON không hợp lệ: {e}")


# ══════════════════════════════════════════════════════════
#  REQUEST MODELS
# ══════════════════════════════════════════════════════════
class AnalyzeURLRequest(BaseModel):
    url: str
    threatScore:  Optional[int]  = 0
    threats:      Optional[List] = []
    hasGambling:  Optional[bool] = False
    vtMalicious:  Optional[int]  = 0
    vtSuspicious: Optional[int]  = 0
    country:      Optional[str]  = None
    org:          Optional[str]  = None


class AnalyzeTextRequest(BaseModel):
    text:  str
    title: Optional[str] = ""


class ProxyAIRequest(BaseModel):
    messages:   List
    max_tokens: Optional[int] = 1000


# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/health")
async def health() -> Dict:
    return {
        "status":      "ok",
        "gemini_keys": len(GEMINI_KEYS),
        "virustotal":  bool(VT_KEY),
        "live_models": _live_models(),
        "cache_size":  len(_cache),
    }


@app.post("/analyze/url")
async def analyze_url(req: AnalyzeURLRequest) -> Dict:
    ck = _cache_key("url", req.url, req.threatScore, req.vtMalicious, req.hasGambling)
    cached = _cache_get(ck)
    if cached is not None:
        return dict(cached, _cached=True)

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

    raw    = await _gemini_call(prompt, 900)
    data   = _parse_json(raw)
    result = dict({"ok": True}, **data)
    _cache_set(ck, result)
    return result


@app.post("/analyze/text")
async def analyze_text(req: AnalyzeTextRequest) -> Dict:
    combined = f"{req.title}\n\n{req.text}".strip()[:3000]
    ck = _cache_key("text", combined)
    cached = _cache_get(ck)
    if cached is not None:
        return dict(cached, _cached=True)

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

    raw    = await _gemini_call(prompt, 800)
    data   = _parse_json(raw)
    result = dict({"ok": True}, **data)
    _cache_set(ck, result)
    return result


@app.post("/proxy/ai")
async def proxy_ai(req: ProxyAIRequest) -> Dict:
    prompt = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            prompt = m.get("content", "")
            break
    if not prompt:
        raise HTTPException(400, detail="Không có nội dung user message")

    text = await _gemini_call(prompt, min(req.max_tokens, 1500))
    return {"ok": True, "text": text}


@app.post("/proxy/dns")
async def proxy_dns(body: Dict) -> Dict:
    domain = body.get("domain", "").strip()
    qtype  = body.get("type", "A")
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    ck = _cache_key("dns", domain, qtype)
    cached = _cache_get(ck)
    if cached is not None:
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
            result: Dict = {
                "ok":      data.get("Status") == 0,
                "records": records,
                "answers": data.get("Answer") or [],
                "domain":  domain,
                "type":    qtype,
            }
        except Exception as e:
            log.warning("DNS error %s: %s", domain, e)
            result = {"ok": False, "records": [], "domain": domain, "type": qtype}

    _cache_set(ck, result, ttl=60)
    return result


CF_PREFIXES = (
    "172.64.", "172.65.", "172.66.", "172.67.", "172.68.", "172.69.",
    "172.70.", "172.71.", "172.72.", "172.73.", "172.74.", "172.75.",
    "172.76.", "172.77.", "172.78.", "172.79.", "172.80.", "172.81.",
    "172.82.", "172.83.", "172.84.", "172.85.", "172.86.", "172.87.",
    "172.88.", "172.89.", "172.90.", "172.91.", "172.92.", "172.93.",
    "172.94.", "172.95.", "172.96.", "172.97.", "172.98.", "172.99.",
    "172.100.", "172.101.", "172.102.", "172.103.", "172.104.", "172.105.",
    "172.106.", "172.107.", "172.108.", "172.109.", "172.110.", "172.111.",
    "172.112.", "172.113.", "172.114.", "172.115.", "172.116.", "172.117.",
    "172.118.", "172.119.", "172.120.", "172.121.", "172.122.", "172.123.",
    "172.124.", "172.125.", "172.126.", "172.127.", "172.128.", "172.129.",
    "172.130.", "172.131.",
    "104.16.", "104.17.", "104.18.", "104.19.", "104.20.", "104.21.",
    "104.22.", "104.23.", "104.24.", "104.25.", "104.26.", "104.27.",
    "104.28.", "104.29.", "104.30.", "104.31.",
    "162.158.", "141.101.64.", "141.101.65.", "141.101.66.", "141.101.67.",
    "188.114.96.", "188.114.97.", "190.93.240.", "198.41.128.",
)


def _is_cloudflare(ip: str) -> bool:
    return any(ip.startswith(p) for p in CF_PREFIXES)


@app.post("/proxy/ipinfo")
async def proxy_ipinfo(body: Dict) -> Dict:
    domain = body.get("domain", "").strip()
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    ck = _cache_key("ipinfo", domain)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    ip = domain
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            dns_res = await client.get(
                "https://dns.google/resolve",
                params={"name": domain, "type": "A"},
                headers={"Accept": "application/dns-json"},
            )
            answers = dns_res.json().get("Answer") or []
            a_recs  = [a["data"] for a in answers if a.get("type") == 1]
            if a_recs:
                ip = a_recs[0]
        except Exception:
            pass

        if _is_cloudflare(ip):
            result: Dict = {
                "ok": True, "ip": ip,
                "country": "Cloudflare CDN", "countryCode": "CF",
                "region": "", "city": "",
                "org": "Cloudflare, Inc.", "isp": "Cloudflare",
                "asn": "AS13335", "isVPN": False, "isTor": False, "isHosting": True,
                "_note": "Cloudflare CDN — không phản ánh server thật",
            }
            _cache_set(ck, result, ttl=3600)
            return result

        try:
            res = await client.get(
                f"https://ip-api.com/json/{ip}",
                params={"fields": "status,country,countryCode,regionName,city,org,isp,as,proxy,hosting,query"},
            )
            if res.is_success:
                d = res.json()
                if d.get("status") == "success":
                    result = {
                        "ok": True,
                        "ip":          d.get("query", ip),
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
            log.warning("ip-api error %s: %s", ip, e)

    return {"ok": False, "ip": ip}


@app.post("/proxy/virustotal/url")
async def proxy_vt_url(body: Dict) -> Dict:
    if not VT_KEY:
        return {"ok": False, "noKey": True}
    url = body.get("url", "")
    if not url:
        raise HTTPException(400, detail="Thiếu url")

    ck = _cache_key("vt_url", url)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

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


@app.post("/proxy/virustotal/domain")
async def proxy_vt_domain(body: Dict) -> Dict:
    if not VT_KEY:
        return {"ok": False, "noKey": True}
    domain = body.get("domain", "")
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    ck = _cache_key("vt_domain", domain)
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(
            f"https://www.virustotal.com/api/v3/domains/{domain}",
            headers={"x-apikey": VT_KEY},
        )
    if not res.is_success:
        return {"ok": False}

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


@app.post("/proxy/urlscan")
async def proxy_urlscan(body: Dict) -> Dict:
    domain = body.get("domain", "")
    ck = _cache_key("urlscan", domain)
    cached = _cache_get(ck)
    if cached is not None:
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
            tags: List[str] = list({
                t
                for r in results
                for t in r.get("verdicts", {}).get("overall", {}).get("tags", [])
            })
            server = results[0].get("page", {}).get("server") if results else None
            result: Dict = {"ok": True, "malicious": is_mal, "tags": tags, "server": server}
            _cache_set(ck, result, ttl=300)
            return result
        except Exception as e:
            log.warning("urlscan error %s: %s", domain, e)
            return {"ok": False, "malicious": False, "tags": []}


@app.post("/proxy/allorigins")
async def proxy_allorigins(body: Dict) -> Dict:
    url = body.get("url", "")
    if not url:
        raise HTTPException(400, detail="Thiếu url")
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        try:
            res = await client.get("https://api.allorigins.win/raw", params={"url": url})
            if res.is_success:
                return {"ok": True, "html": res.text[:300_000]}
        except Exception as e:
            log.warning("allorigins error %s: %s", url, e)
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
        workers=1,
    )
