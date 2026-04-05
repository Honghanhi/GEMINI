"""
AI-PROOF Backend  —  FastAPI + Gemini API
Deploy: Render.com (Free tier — ~80MB RAM, không load model local)

ENV VARS cần set trên Render:
  GEMINI_API_KEY=your_key_here        ← bắt buộc
  VIRUSTOTAL_API_KEY=your_key_here    ← tuỳ chọn
  ALLOWED_ORIGINS=https://your-site.com  ← tuỳ chọn, mặc định *
"""

import os, json, asyncio, hashlib, time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")
VT_KEY       = os.getenv("VIRUSTOTAL_API_KEY", "")
ORIGINS      = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Gemini models theo thứ tự ưu tiên (thử lần lượt khi 429/404)
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]

TIMEOUT = httpx.Timeout(30.0, connect=10.0)

app = FastAPI(title="AI-PROOF Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # frontend gọi được
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limit state (per-process, đủ dùng với 1 worker) ──
_rl_until   = 0.0   # timestamp: không gọi Gemini trước thời điểm này
_rl_model_i = 0     # index model hiện tại đang dùng

# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

async def _gemini_call(prompt: str, max_tokens: int = 1000) -> str:
    """Gọi Gemini với auto-fallback model + backoff 429."""
    global _rl_until, _rl_model_i

    if not GEMINI_KEY:
        raise HTTPException(503, detail="GEMINI_API_KEY chưa cấu hình")

    # Nếu đang trong cooldown → chờ
    wait = _rl_until - time.time()
    if wait > 0:
        if wait > 120:
            raise HTTPException(429, detail=f"Quota Gemini đang reset, thử lại sau {int(wait)}s")
        await asyncio.sleep(wait)

    tried = set()
    idx   = _rl_model_i

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(len(GEMINI_MODELS) * 2):  # tối đa 2 vòng
            model = GEMINI_MODELS[idx % len(GEMINI_MODELS)]

            if model in tried:
                # đã thử hết, chờ 60s rồi thử lại model đầu
                _rl_until = time.time() + 60
                _rl_model_i = 0
                await asyncio.sleep(60)
                idx = 0
                tried.clear()

            tried.add(model)
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={GEMINI_KEY}"
            )
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
            }

            try:
                res = await client.post(url, json=body)
            except httpx.TimeoutException:
                idx += 1
                continue

            if res.status_code == 200:
                data = res.json()
                text = (data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", ""))
                _rl_model_i = idx % len(GEMINI_MODELS)  # lưu model đang dùng tốt
                return text

            if res.status_code == 429:
                retry_after = int(res.headers.get("Retry-After", "30"))
                wait_sec    = max(retry_after, 30)
                print(f"[Gemini] 429 {model} → chờ {wait_sec}s rồi thử model tiếp")
                _rl_until = time.time() + wait_sec
                await asyncio.sleep(wait_sec)
                idx += 1
                continue

            if res.status_code == 404:
                print(f"[Gemini] 404 {model} → thử model khác")
                idx += 1
                continue

            # Lỗi khác (400, 403) → dừng luôn
            raise HTTPException(res.status_code, detail=f"Gemini error: {res.text[:200]}")

    raise HTTPException(503, detail="Tất cả Gemini models đều thất bại")


def _parse_json(text: str) -> dict:
    """Parse JSON từ text Gemini (bỏ markdown fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    # tìm object JSON đầu tiên
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return json.loads(text)


# ══════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════

class AnalyzeURLRequest(BaseModel):
    url: str
    threatScore: Optional[int] = 0
    threats: Optional[list]    = []
    hasGambling: Optional[bool]= False
    vtMalicious: Optional[int] = 0
    vtSuspicious:Optional[int] = 0
    country: Optional[str]     = None
    org: Optional[str]         = None

class AnalyzeTextRequest(BaseModel):
    text: str
    title: Optional[str] = ""

class ProxyAIRequest(BaseModel):
    messages: list
    max_tokens: Optional[int] = 1000

# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gemini": bool(GEMINI_KEY),
        "virustotal": bool(VT_KEY),
        "models": GEMINI_MODELS,
    }


# ── /analyze/url — phân tích URL cờ bạc / độc hại ────────
@app.post("/analyze/url")
async def analyze_url(req: AnalyzeURLRequest):
    prompt = f"""Bạn là chuyên gia an ninh mạng Việt Nam. Phân tích URL sau dựa trên dữ liệu được cung cấp.

URL: {req.url}
Threat Score hiện tại: {req.threatScore}/100
VirusTotal: {req.vtMalicious} độc hại, {req.vtSuspicious} nghi ngờ
Gambling phát hiện: {"CÓ" if req.hasGambling else "Không"}
Quốc gia server: {req.country or "Không rõ"}
ISP/Org: {req.org or "Không rõ"}
Các mối đe dọa: {", ".join(req.threats) if req.threats else "Không có"}

Trả về JSON (KHÔNG markdown, KHÔNG text ngoài JSON):
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
  "isVietnamese": true/false,
  "vietnameseNotes": "ghi chú nếu là web VN hoặc nhắm vào người VN"
}}"""

    raw  = await _gemini_call(prompt, 900)
    data = _parse_json(raw)
    return {"ok": True, **data}


# ── /analyze/text — phân tích tin giả / fake news ────────
@app.post("/analyze/text")
async def analyze_text(req: AnalyzeTextRequest):
    combined = f"{req.title}\n\n{req.text}".strip()[:3000]
    prompt = f"""Bạn là fact-checker chuyên nghiệp. Phân tích nội dung sau:

{combined}

Trả về JSON (KHÔNG markdown):
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
    return {"ok": True, **data}


# ── /proxy/ai — general AI proxy (tương thích code cũ) ───
@app.post("/proxy/ai")
async def proxy_ai(req: ProxyAIRequest):
    # Lấy message cuối cùng của user
    prompt = ""
    for m in req.messages:
        if m.get("role") == "user":
            prompt = m.get("content", "")
    if not prompt:
        raise HTTPException(400, detail="Không có nội dung")

    text = await _gemini_call(prompt, min(req.max_tokens, 1500))
    return {"ok": True, "text": text}


# ── /proxy/dns — DNS over HTTPS ──────────────────────────
@app.post("/proxy/dns")
async def proxy_dns(body: dict):
    domain = body.get("domain", "")
    qtype  = body.get("type", "A")
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(
            "https://dns.google/resolve",
            params={"name": domain, "type": qtype},
            headers={"Accept": "application/dns-json"},
        )
    if not res.is_success:
        return {"ok": False, "records": [], "domain": domain, "type": qtype}

    data    = res.json()
    records = [a["data"] for a in (data.get("Answer") or []) if a.get("type") == 1]
    return {
        "ok":      data.get("Status") == 0,
        "records": records,
        "answers": data.get("Answer") or [],
        "domain":  domain,
        "type":    qtype,
    }


# ── /proxy/ipinfo — IP geolocation ───────────────────────
CF_PREFIXES = [
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
]

def _is_cloudflare(ip: str) -> bool:
    return any(ip.startswith(p) for p in CF_PREFIXES)

@app.post("/proxy/ipinfo")
async def proxy_ipinfo(body: dict):
    domain = body.get("domain", "")
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    # Resolve IP
    ip = domain
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            dns_res = await client.get(
                "https://dns.google/resolve",
                params={"name": domain, "type": "A"},
                headers={"Accept": "application/dns-json"},
            )
            answers = dns_res.json().get("Answer") or []
            a_records = [a["data"] for a in answers if a.get("type") == 1]
            if a_records:
                ip = a_records[0]
        except Exception:
            pass

    # Cloudflare → trả ngay
    if _is_cloudflare(ip):
        return {
            "ok": True, "ip": ip,
            "country": "Cloudflare CDN", "countryCode": "CF",
            "region": "", "city": "",
            "org": "Cloudflare, Inc.", "isp": "Cloudflare",
            "asn": "AS13335", "isVPN": False, "isTor": False, "isHosting": True,
            "_note": "Cloudflare CDN — không phản ánh server thật",
        }

    # ip-api.com (backend gọi không bị 403)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            res = await client.get(
                f"https://ip-api.com/json/{ip}",
                params={"fields": "status,country,countryCode,regionName,city,org,isp,as,proxy,hosting,query"},
            )
            if res.is_success:
                d = res.json()
                if d.get("status") == "success":
                    return {
                        "ok": True, "ip": d.get("query", ip),
                        "country": d.get("country"), "countryCode": d.get("countryCode"),
                        "region": d.get("regionName"), "city": d.get("city"),
                        "org": d.get("org"), "isp": d.get("isp"),
                        "asn": d.get("as"), "isVPN": d.get("proxy", False),
                        "isTor": False, "isHosting": d.get("hosting", False),
                    }
        except Exception:
            pass

    return {"ok": False, "ip": ip}


# ── /proxy/virustotal/url ─────────────────────────────────
@app.post("/proxy/virustotal/url")
async def proxy_vt_url(body: dict):
    if not VT_KEY:
        return {"ok": False, "noKey": True}
    url = body.get("url", "")
    if not url:
        raise HTTPException(400, detail="Thiếu url")

    encoded = __import__("base64").urlsafe_b64encode(url.encode()).rstrip(b"=").decode()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(
            f"https://www.virustotal.com/api/v3/urls/{encoded}",
            headers={"x-apikey": VT_KEY},
        )
        if not res.is_success:
            # Submit mới
            sub = await client.post(
                "https://www.virustotal.com/api/v3/urls",
                headers={"x-apikey": VT_KEY, "Content-Type": "application/x-www-form-urlencoded"},
                content=f"url={url}",
            )
            if not sub.is_success:
                return {"ok": False}
            return {"ok": True, "malicious": 0, "suspicious": 0, "totalEngines": 0, "submitted": True}

    stats = res.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    return {
        "ok": True,
        "malicious":    stats.get("malicious", 0),
        "suspicious":   stats.get("suspicious", 0),
        "harmless":     stats.get("harmless", 0),
        "undetected":   stats.get("undetected", 0),
        "totalEngines": sum(stats.values()),
    }


# ── /proxy/virustotal/domain ──────────────────────────────
@app.post("/proxy/virustotal/domain")
async def proxy_vt_domain(body: dict):
    if not VT_KEY:
        return {"ok": False, "noKey": True}
    domain = body.get("domain", "")
    if not domain:
        raise HTTPException(400, detail="Thiếu domain")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(
            f"https://www.virustotal.com/api/v3/domains/{domain}",
            headers={"x-apikey": VT_KEY},
        )
    if not res.is_success:
        return {"ok": False}

    attr = res.json().get("data", {}).get("attributes", {})
    cd   = attr.get("creation_date")
    return {
        "ok":           True,
        "reputation":   attr.get("reputation", 0),
        "categories":   attr.get("categories", {}),
        "country":      attr.get("country"),
        "creationDate": __import__("datetime").datetime.utcfromtimestamp(cd).strftime("%Y-%m-%d") if cd else None,
    }


# ── /proxy/urlscan ────────────────────────────────────────
@app.post("/proxy/urlscan")
async def proxy_urlscan(body: dict):
    domain = body.get("domain", "")
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
            is_mal  = any(r.get("verdicts", {}).get("overall", {}).get("malicious") for r in results)
            tags    = list({t for r in results for t in r.get("verdicts", {}).get("overall", {}).get("tags", [])})
            server  = results[0].get("page", {}).get("server") if results else None
            return {"ok": True, "malicious": is_mal, "tags": tags, "server": server}
        except Exception:
            return {"ok": False, "malicious": False, "tags": []}


# ── /proxy/allorigins — fetch HTML qua proxy ─────────────
@app.post("/proxy/allorigins")
async def proxy_allorigins(body: dict):
    url = body.get("url", "")
    if not url:
        raise HTTPException(400, detail="Thiếu url")
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        try:
            res = await client.get(
                "https://api.allorigins.win/raw",
                params={"url": url},
            )
            if res.is_success:
                return {"ok": True, "html": res.text[:300000]}
        except Exception:
            pass
    return {"ok": False, "html": ""}


# ══════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
