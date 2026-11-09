"""Public CiteRAG Lite demo for the Northflank free Sandbox plan.

The free plan provides 0.2 vCPU / 512 MB, which cannot host the PyTorch +
MiniLM + LanceDB semantic stack. This entrypoint therefore serves the same
dashboard and REST shape using CiteRAG's documented standard-library SQLite
FTS5 fallback (``citerag.cli``). Retrieval is keyword-based, but every answer
still carries a source path and a verbatim excerpt.

The production entrypoint (``python -m citerag.web``) refuses a non-loopback
listener without a bearer token, and the dashboard cannot send a bearer token
from a browser. This entrypoint deliberately serves a public, read-only demo of
public project documentation. It must not be used with private data, provider
credentials, or production traffic.
"""
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DB = ROOT / "data/index.sqlite3"
ABSTENTION = "Insufficient evidence in the approved project sources."
STOP = set(
    "a an the is are was were what which where when how does do did this that "
    "project projects use used uses please about in on of and or to for with "
    "from can could would should be been being me my our your show find tell "
    "give explain".split()
)

_jobs = {}
_guard = threading.Lock()
_generation = threading.Lock()


def _terms(query):
    return list(
        dict.fromkeys(
            term.lower()
            for term in re.findall(r"\w+", query)
            if len(term) > 1 and term.lower() not in STOP
        )
    )[:32]


def _snippet(content, terms, width=1200):
    lower = content.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
    start = max(0, (min(positions) if positions else 0) - 240)
    end = min(len(content), start + width)
    return ("…" if start else "") + content[start:end] + ("…" if end < len(content) else "")


def search(query, limit=3):
    """Return bounded FTS5 matches with source paths and verbatim excerpts."""
    terms = _terms(query)
    if not terms or not DB.is_file():
        return []
    expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    con = sqlite3.connect("file:" + str(DB) + "?mode=ro", uri=True, timeout=3)
    try:
        rows = con.execute(
            "SELECT path, content FROM documents WHERE documents MATCH ? "
            "ORDER BY bm25(documents) LIMIT ?",
            (expression, max(1, min(int(limit), 10))),
        ).fetchall()
    except sqlite3.DatabaseError:
        return []
    finally:
        con.close()
    return [
        {"source": path, "id": secrets.token_hex(8), "text": _snippet(content, terms)}
        for path, content in rows
    ]


def _indexed_documents():
    if not DB.is_file():
        return 0
    con = sqlite3.connect("file:" + str(DB) + "?mode=ro", uri=True, timeout=3)
    try:
        return con.execute("SELECT count(*) FROM documents").fetchone()[0]
    except sqlite3.DatabaseError:
        return 0
    finally:
        con.close()


def _iso_mtime(path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def _demo_html():
    html = (Path(__file__).with_name("web.html")).read_text()
    banner = (
        '<div class="panel" style="border-color:#6f8f4f">'
        "<strong>CiteRAG Lite demo</strong> — SQLite FTS5 keyword retrieval over the "
        "public project documentation. The full MiniLM/LanceDB semantic backend needs "
        "more memory than the Northflank free plan provides (0.2 vCPU / 512 MB)."
        "</div>"
    )
    html = html.replace("<main>", "<main>" + banner, 1)
    html = html.replace(
        "LanceDB · all-MiniLM-L6-v2 · OmniRoute (optional) · CPU retrieval",
        "SQLite FTS5 fallback · public demo · CPU retrieval",
    )
    return html


def _adapters():
    return [
        {
            "name": "retrieval-only",
            "schema": {"type": "object", "additionalProperties": False},
            "generation": False,
            "available": True,
            "status": "ready",
        },
        {
            "name": "omniroute",
            "schema": {"type": "object", "additionalProperties": False},
            "generation": True,
            "available": False,
            "status": "not_configured",
        },
        {
            "name": "opencode",
            "schema": {"type": "object", "additionalProperties": False},
            "generation": True,
            "available": False,
            "status": "not_configured",
        },
    ]


def run_job(job_id, question):
    started = time.monotonic()
    try:
        rows = search(question, 3)
        if rows:
            top = rows[0]
            answer = top["text"] + "\n\nSource: " + top["source"]
            result = {
                "status": "completed",
                "answer": answer,
                "sources": [row["source"] for row in rows],
                "mode": "retrieval_only",
                "citations_verified": True,
                "claims_supported": True,
                "evidence": rows,
            }
        else:
            result = {
                "status": "insufficient_evidence",
                "answer": ABSTENTION,
                "sources": [],
                "mode": "retrieval_only",
                "evidence": [],
            }
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        result["state"] = "completed"
    except Exception as exc:  # keep the demo process alive; do not leak details
        result = {"state": "failed", "status": "failed", "error": type(exc).__name__}
    finally:
        _generation.release()
    with _guard:
        _jobs[job_id] = result


def run_reindex(job_id):
    try:
        from citerag.cli import ingest

        count = ingest(DB, DOCS)
        result = {"state": "completed", "status": "completed", "documents": count}
    except Exception as exc:
        result = {"state": "failed", "status": "failed", "error": type(exc).__name__}
    finally:
        _generation.release()
    with _guard:
        _jobs[job_id] = result


def _start_job(target, *args):
    if not _generation.acquire(blocking=False):
        return None
    job_id = secrets.token_urlsafe(18)
    with _guard:
        if len(_jobs) >= 20:
            for old in list(_jobs):
                if _jobs[old].get("state") != "running":
                    del _jobs[old]
                    break
        _jobs[job_id] = {"state": "running"}
    return JSONResponse(
        {"job_id": job_id, "state": "running"},
        status_code=202,
        background=BackgroundTask(target, job_id, *args),
    ), job_id


async def home(request):
    return HTMLResponse(await run_in_threadpool(_demo_html))


async def health(request):
    return JSONResponse(
        {
            "status": "ok",
            "status_scope": "HTTP process liveness only",
            "retrieval_index": "present" if DB.is_file() else "uninitialized",
            "retrieval_backend": "sqlite-fts5",
            "generation": "not_configured",
            "model": None,
            "native_opencode": "not_configured",
            "busy": _generation.locked(),
        }
    )


async def adapters(request):
    return JSONResponse({"adapters": _adapters()})


async def context_status(request):
    count = await run_in_threadpool(_indexed_documents)
    attempt_path = ROOT / "data/context-refresh-attempt.json"
    attempt = json.loads(attempt_path.read_text()) if attempt_path.exists() else {}
    return JSONResponse(
        {
            "workspace": "demo",
            "repository_sources": [],
            "admission_limits": {},
            "imported_documents": count,
            "selected_sessions": 0,
            "imported_messages": 0,
            "indexed_chunks": count,
            "last_sync": _iso_mtime(DB),
            "freshness": attempt,
            "last_successful_ingestion": _iso_mtime(DB),
            "phase": "demo",
            "coverage": "Public project documents under docs/; SQLite FTS5 keyword retrieval",
            "native_opencode": "not_configured",
            "native_answers": 0,
        }
    )


async def sources(request):
    return JSONResponse(
        {"workspace": "demo", "sources": [], "management": "operator-configured"}
    )


async def reindex(request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected an empty JSON object"}, status_code=400)
    if data != {}:
        return JSONResponse({"error": "Sources are operator-configured"}, status_code=400)
    response = _start_job(run_reindex)
    if response is None:
        return JSONResponse({"error": "A request is already running"}, status_code=409)
    return response[0]


async def ask(request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected JSON"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "Expected JSON object"}, status_code=400)
    question = data.get("question")
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
        return JSONResponse(
            {"error": "Enter a question between 1 and 4000 characters."},
            status_code=400,
        )
    if data.get("adapter_config"):
        return JSONResponse({"error": "Invalid adapter config"}, status_code=400)
    adapter = data.get("adapter")
    if adapter not in (None, "retrieval-only"):
        return JSONResponse(
            {"error": "Adapter is unavailable or unverified"}, status_code=503
        )
    if data.get("use_native") is True:
        return JSONResponse(
            {"error": "Adapter is unavailable or unverified"}, status_code=503
        )
    response = _start_job(run_job, question.strip())
    if response is None:
        return JSONResponse(
            {"error": "A request is already running. Wait for it to finish."},
            status_code=409,
        )
    return response[0]


async def job(request):
    with _guard:
        result = _jobs.get(request.path_params["job_id"])
    return JSONResponse(
        result if result is not None else {"error": "Unknown request"},
        status_code=200 if result is not None else 404,
    )


app = Starlette(
    routes=[
        Route("/", home),
        Route("/api/health", health),
        Route("/api/adapters", adapters),
        Route("/api/context", context_status),
        Route("/api/sources", sources),
        Route("/api/reindex", reindex, methods=["POST"]),
        Route("/api/ask", ask, methods=["POST"]),
        Route("/api/jobs/{job_id}", job),
    ]
)


def validate_demo_configuration() -> None:
    """Fail closed unless the operator explicitly opted into the public demo."""
    if os.getenv("CITERAG_DEMO_PUBLIC") != "1":
        raise SystemExit("CITERAG_DEMO_PUBLIC=1 is required for the demo entrypoint")
    if os.getenv("CITERAG_PRODUCTION") == "1" or os.getenv("CITERAG_REQUIRE_AUTH") == "1":
        raise SystemExit("The demo entrypoint cannot run in production/auth mode")
    if os.getenv("CITERAG_API_TOKEN"):
        raise SystemExit("The demo entrypoint does not accept a bearer token")


def main() -> None:
    validate_demo_configuration()
    if not DB.is_file():
        raise SystemExit("Demo index is missing; build the image with citerag.cli ingest")
    uvicorn.run(
        app,
        host=os.getenv("CITERAG_WEB_HOST", "0.0.0.0"),
        port=int(os.getenv("CITERAG_WEB_PORT", "18762")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
