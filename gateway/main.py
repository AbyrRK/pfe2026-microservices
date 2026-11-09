"""
API Gateway: the ONLY thing the company chatbot talks to.
Routes:
  /api/v1/documents/*  -> Document Service
  /api/v1/chat/*       -> Q&A Service (fast path)
  /api/v1/research/*   -> Q&A Service (deep path)
"""
import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

DOCUMENT_SERVICE_URL = os.getenv("DOCUMENT_SERVICE_URL", "http://document-service:8001")
QA_SERVICE_URL = os.getenv("QA_SERVICE_URL", "http://qa-service:8002")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")

# Per-route timeouts (seconds). Uploads of scanned PDFs run OCR synchronously,
# so they need a long limit until uploads are made asynchronous.
DOCUMENTS_TIMEOUT = float(os.getenv("DOCUMENTS_TIMEOUT", "600"))
CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT", "15"))
RESEARCH_TIMEOUT = float(os.getenv("RESEARCH_TIMEOUT", "10"))

# Headers that must not be forwarded: httpx recomputes them for the new request.
_SKIP_HEADERS = {"host", "content-length"}

app = FastAPI(title="PFE2026 Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(request: Request):
    if not GATEWAY_API_KEY:
        return
    if request.headers.get("X-API-Key") != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _forward_headers(request: Request) -> dict:
    return {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_HEADERS}


async def _proxy(request: Request, target_base: str, path: str, timeout: float):
    _check_auth(request)
    url = f"{target_base}{path}"
    body = await request.body()
    headers = _forward_headers(request)

    # Errors are RETURNED (not raised) so they stay inside CORSMiddleware and
    # the browser can read them instead of showing "Failed to fetch".
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
            resp = await client.request(
                method=request.method, url=url, content=body,
                headers=headers, params=request.query_params,
            )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"detail": "The service took too long to respond. "
                               "Large or scanned documents may need more time."},
        )
    except httpx.RequestError as e:
        return JSONResponse(
            status_code=502,
            content={"detail": f"Upstream service unreachable ({type(e).__name__})."},
        )

    try:
        content = resp.json() if resp.content else {}
    except ValueError:
        content = {"detail": resp.text[:500]}
    return JSONResponse(status_code=resp.status_code, content=content)


# ── Streaming route: MUST be registered before the generic chat_proxy ─────────
# /chat/ask/stream is Server-Sent Events, so bytes are streamed through as
# they arrive instead of waiting for a full JSON body.
@app.api_route("/api/v1/chat/ask/stream", methods=["POST"])
async def chat_stream_proxy(request: Request):
    _check_auth(request)
    url = f"{QA_SERVICE_URL}/chat/ask/stream"
    body = await request.body()
    headers = _forward_headers(request)

    async def event_generator():
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=5.0)
            ) as client:
                async with client.stream("POST", url, content=body, headers=headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except httpx.RequestError as e:
            msg = f"Upstream service unreachable ({type(e).__name__})."
            yield f'data: {{"error": "{msg}"}}\n\n'.encode()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.api_route("/api/v1/documents/{path:path}", methods=["GET", "POST", "DELETE"])
async def documents_proxy(path: str, request: Request):
    return await _proxy(request, DOCUMENT_SERVICE_URL, f"/documents/{path}", DOCUMENTS_TIMEOUT)


@app.api_route("/api/v1/chat/{path:path}", methods=["GET", "POST", "DELETE"])
async def chat_proxy(path: str, request: Request):
    return await _proxy(request, QA_SERVICE_URL, f"/chat/{path}", CHAT_TIMEOUT)


@app.api_route("/api/v1/research/{path:path}", methods=["GET", "POST"])
async def research_proxy(path: str, request: Request):
    return await _proxy(request, QA_SERVICE_URL, f"/research/{path}", RESEARCH_TIMEOUT)


@app.get("/health")
async def health():
    status = {"gateway": "ok"}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, base in [("document_service", DOCUMENT_SERVICE_URL),
                           ("qa_service", QA_SERVICE_URL)]:
            try:
                r = await client.get(f"{base}/health")
                status[name] = r.json()
            except Exception as e:
                status[name] = f"unreachable: {e}"
    return status


# ── Legacy routes (without /api/v1 prefix) for frontend compatibility ─────────
@app.api_route("/chat/ask/stream", methods=["POST"], include_in_schema=False)
async def chat_stream_legacy(request: Request):
    return await chat_stream_proxy(request)


@app.api_route("/documents/{path:path}", methods=["GET", "POST", "DELETE"],
               include_in_schema=False)
async def documents_legacy(path: str, request: Request):
    return await documents_proxy(path, request)


@app.api_route("/chat/{path:path}", methods=["GET", "POST", "DELETE"],
               include_in_schema=False)
async def chat_legacy(path: str, request: Request):
    return await chat_proxy(path, request)


@app.api_route("/research/{path:path}", methods=["GET", "POST", "DELETE"],
               include_in_schema=False)
async def research_legacy(path: str, request: Request):
    return await research_proxy(path, request)