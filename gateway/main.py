"""
API Gateway — the ONLY thing the company chatbot talks to.
Routes:
  /api/v1/documents/*  -> Document Service
  /api/v1/chat/*        -> Q&A Service (fast path)
  /api/v1/research/*    -> Q&A Service (deep path)
"""
import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

DOCUMENT_SERVICE_URL = os.getenv("DOCUMENT_SERVICE_URL", "http://document-service:8001")
QA_SERVICE_URL = os.getenv("QA_SERVICE_URL", "http://qa-service:8002")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")

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


async def _proxy(request: Request, target_base: str, path: str, timeout: float):
    _check_auth(request)
    url = f"{target_base}{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(
            method=request.method, url=url, content=body,
            headers=headers, params=request.query_params,
        )

    return JSONResponse(status_code=resp.status_code, content=resp.json() if resp.content else {})


# ── Streaming route — MUST be registered before the generic chat_proxy ────────
# /chat/ask/stream is Server-Sent Events, not JSON — the generic _proxy()
# waits for a full JSON body that never arrives this way, causing the
# frontend's fetch() to hang and fail. This route streams bytes through
# as they arrive instead.
@app.api_route("/api/v1/chat/ask/stream", methods=["POST"])
async def chat_stream_proxy(request: Request):
    _check_auth(request)
    url = f"{QA_SERVICE_URL}/chat/ask/stream"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    async def event_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, content=body, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.api_route("/api/v1/documents/{path:path}", methods=["GET", "POST", "DELETE"])
async def documents_proxy(path: str, request: Request):
    return await _proxy(request, DOCUMENT_SERVICE_URL, f"/documents/{path}", 60.0)


@app.api_route("/api/v1/chat/{path:path}", methods=["GET", "POST", "DELETE"])
async def chat_proxy(path: str, request: Request):
    return await _proxy(request, QA_SERVICE_URL, f"/chat/{path}", 15.0)


@app.api_route("/api/v1/research/{path:path}", methods=["GET", "POST"])
async def research_proxy(path: str, request: Request):
    return await _proxy(request, QA_SERVICE_URL, f"/research/{path}", 10.0)


@app.get("/health")
async def health():
    status = {"gateway": "ok"}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, base in [("document_service", DOCUMENT_SERVICE_URL), ("qa_service", QA_SERVICE_URL)]:
            try:
                r = await client.get(f"{base}/health")
                status[name] = r.json()
            except Exception as e:
                status[name] = f"unreachable: {e}"
    return status


# ── Legacy routes (without /api/v1/ prefix) for frontend compatibility ────────
# Also needs the same streaming fix — same problem, same reason.
@app.api_route("/chat/ask/stream", methods=["POST"], include_in_schema=False)
async def chat_stream_legacy(request: Request):
    return await chat_stream_proxy(request)


@app.api_route(
    "/documents/{path:path}",
    methods=["GET", "POST", "DELETE"],
    include_in_schema=False,
)
async def documents_legacy(path: str, request: Request):
    return await documents_proxy(path, request)

@app.api_route(
    "/chat/{path:path}",
    methods=["GET", "POST", "DELETE"],
    include_in_schema=False,
)
async def chat_legacy(path: str, request: Request):
    return await chat_proxy(path, request)

@app.api_route(
    "/research/{path:path}",
    methods=["GET", "POST", "DELETE"],
    include_in_schema=False,
)
async def research_legacy(path: str, request: Request):
    return await research_proxy(path, request)