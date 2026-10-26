"""HTTP routes and limits; inference stays behind pipeline interfaces."""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException
from donations.store import CONSENT_TEXT, CONSENT_VERSION, CapacityReached, DonationStore
from .performance import performance
from pipeline.bundle import load_bundle
from pipeline.decision import OUTCOMES
from pipeline.preprocess import CannotAnalyze
from pipeline.service import DISPLAY_VERSION, AnalysisWorker
from .settings import Settings

logger = logging.getLogger("speech.metadata")
storage_logger = logging.getLogger("speech.storage")


class DonationRequest(BaseModel):
    token: str = Field(min_length=40, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    consent_version: str
    confirm: bool
    eligibility_confirmed: bool = False


class BodyLimit:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                # Starlette closes partial multipart spools on MultiPartException.
                raise MultiPartException("request_body_too_large")
            return message
        await self.app(scope, bounded_receive, send)


def create_app(settings=None, *, worker_factory=AnalysisWorker):
    settings = settings or Settings.from_env()
    if not 1 <= settings.concurrency <= 2 or not 1 <= settings.onnx_threads <= 16:
        raise ValueError("Invalid resource settings")
    if settings.max_upload_bytes > 25 * 1024 * 1024 or settings.timeout_seconds <= 0:
        raise ValueError("Invalid upload or timeout limits")
    if (not 1 <= settings.donation_retention_days <= 30 or min(settings.donation_max_bytes, settings.donation_max_files,
            settings.donation_daily_max, settings.donation_hourly_max) <= 0 or settings.donation_min_free_bytes < 0):
        raise ValueError("Invalid donation storage limits")
    if settings.feedback_url and (not settings.feedback_url.startswith("https://") or len(settings.feedback_url) > 300
                                  or any(c in settings.feedback_url for c in "\"'<> ")):
        raise ValueError("FEEDBACK_URL must be a plain https URL")

    def log_storage(event, **counts):
        files, size = app.state.donations.usage()
        storage_logger.info(json.dumps({"event": event, "files": files, "bytes": size, **counts}))

    @asynccontextmanager
    async def lifespan(app):
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        app.state.bundle = load_bundle(settings.active_bundle, root=settings.root, threads=settings.onnx_threads, initialize_detector=False)
        app.state.donations = DonationStore(settings.donation_dir, max_bytes=settings.donation_max_bytes,
            max_files=settings.donation_max_files, daily_max=settings.donation_daily_max, hourly_max=settings.donation_hourly_max,
            min_free_bytes=settings.donation_min_free_bytes, retention_days=settings.donation_retention_days,
            store_audio=settings.donation_store_audio)
        log_storage("startup", **app.state.donations.reconcile(), expired=app.state.donations.expire())
        app.state.workers, app.state.idle_workers = [], []
        app.state.semaphore = asyncio.Semaphore(settings.concurrency)
        try:
            for _ in range(settings.concurrency):
                worker = await asyncio.to_thread(worker_factory, settings.root, settings.active_bundle, settings.onnx_threads, settings.ffmpeg, settings.upload_dir)
                app.state.workers.append(worker)
                app.state.idle_workers.append(worker)

            async def expire_tokens():
                ticks = 0
                while True:
                    await asyncio.sleep(1)
                    app.state.donations.purge()
                    ticks += 1
                    if ticks % 3600 == 0:
                        expired = await asyncio.to_thread(app.state.donations.expire)
                        if expired:
                            log_storage("expire", expired=expired)
            expiry = asyncio.create_task(expire_tokens())
            try:
                yield
            finally:
                expiry.cancel()
        finally:
            for worker in app.state.workers:
                await asyncio.to_thread(worker.close)
            app.state.donations.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit, limit=settings.max_upload_bytes + 1024 * 1024)
    app.state.settings = settings
    app.state.inflight_uploads = 0

    def notice(bundle):
        return {"text": "연구 시연입니다. 서비스 성능이 검증되지 않았으며 점수는 AI 확률이 아닙니다. DEV 기반 보정이며 음성 판정은 휴리스틱입니다. 비상업적 이용만 허용됩니다.",
                "development_rates": bundle.calibration.get("development_rates", {}) if bundle else {},
                "heldout_development_rates": bundle.calibration.get("heldout_development_rates", {}) if bundle else {}, "license": "CC BY-NC-SA 4.0"}

    def response_data(request, reason=None):
        bundle = getattr(app.state, "bundle", None)
        return {"request_id": request.state.request_id, "outcome": OUTCOMES["cannot"], "outcome_code": "cannot", "reason": reason,
                "file_score": None, "windows": [], "slots": [], "duration": None, "bundle_id": bundle.bundle_id if bundle else None,
                "hashes": bundle.hashes if bundle else {}, "thresholds": {k: bundle.calibration[k] for k in ("t_low", "t_warn")} if bundle else {},
                "notice": notice(bundle), "donation_token": None}

    @app.middleware("http")
    async def request_context(request, call_next):
        request.state.request_id = str(uuid.uuid4())
        length = request.headers.get("content-length")
        is_upload = request.method == "POST" and request.url.path == "/api/analyze"
        if length and (not length.isascii() or not length.isdigit() or len(length) > 12 or int(length) > settings.max_upload_bytes + 1024 * 1024):
            result = JSONResponse(response_data(request, "file_too_large"), status_code=413)
        elif is_upload and app.state.inflight_uploads >= 8:
            result = JSONResponse(response_data(request, "server_busy"), status_code=503)
        else:
            if is_upload:
                app.state.inflight_uploads += 1
            try:
                result = await call_next(request)
            except Exception:
                result = JSONResponse(response_data(request, "request_failed"), status_code=500)
            finally:
                if is_upload:
                    app.state.inflight_uploads -= 1
        result.headers.update({"X-Request-ID": request.state.request_id, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                               "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()", "Cross-Origin-Opener-Policy": "same-origin",
                               "Content-Security-Policy": "default-src 'self'; media-src 'self' blob:; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"})
        return result

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(response_data(request, "invalid_request"), status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(response_data(request, "request_body_too_large" if "request_body_too_large" in str(exc.detail) else "invalid_request"), status_code=exc.status_code)

    @app.get("/api/health")
    async def health():
        ready = all(getattr(w, "process", None) is None or w.process.is_alive() for w in app.state.workers)
        return JSONResponse({"status": "ok" if ready else "unhealthy", "bundle_id": app.state.bundle.bundle_id, "hashes": app.state.bundle.hashes}, status_code=200 if ready else 503)

    @app.get("/api/config")
    async def config():
        return {"max_upload_bytes": settings.max_upload_bytes, "max_duration_seconds": 120, "donation_token_seconds": 900,
                "consent_version": CONSENT_VERSION, "consent_text": CONSENT_TEXT, "bundle_id": app.state.bundle.bundle_id, "notice": notice(app.state.bundle),
                "partial_enabled": app.state.bundle.partial is not None,
                "partial_scope": (app.state.bundle.partial or {}).get("scope_ko"), "display_version": DISPLAY_VERSION,
                "donations_enabled": settings.donations_enabled, "donation_retention_days": settings.donation_retention_days,
                "donation_store_audio": settings.donation_store_audio,
                "feedback_url": settings.feedback_url or None,
                "performance": performance(app.state.bundle.partial is not None)}

    @app.post("/api/analyze")
    async def analyze(request: Request, file: UploadFile = File(...), analysis_consent: bool = Form(False), donation_opt_in: bool = Form(False)):
        started, body, worker = time.perf_counter(), bytearray(), None
        result = response_data(request)
        try:
            if not analysis_consent:
                raise CannotAnalyze("analysis_consent_required")
            if Path(file.filename or "").suffix.lower() not in {".wav", ".mp3", ".flac"}:
                raise CannotAnalyze("unsupported_format")
            while chunk := await file.read(1024 * 1024):
                body.extend(chunk)
                if len(body) > settings.max_upload_bytes:
                    raise CannotAnalyze("file_too_large")
            await file.close()
            try:
                await asyncio.wait_for(app.state.semaphore.acquire(), timeout=settings.queue_timeout_seconds)
                worker = app.state.idle_workers.pop()
            except asyncio.TimeoutError:
                raise CannotAnalyze("server_busy")
            task = asyncio.create_task(asyncio.to_thread(worker.analyze, bytes(body), settings.timeout_seconds))
            try:
                scored = await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            result.update(scored)
            result["reason"] = None
            if donation_opt_in and settings.donations_enabled:
                try:
                    app.state.donations.check_capacity(len(body))
                    result["donation_token"] = app.state.donations.offer(body, result["outcome_code"], app.state.bundle.bundle_id)
                    if result["donation_token"] is None:
                        result["donation_reason"] = "pending_capacity_reached"
                except CapacityReached:
                    result["donation_reason"] = "donation_capacity_reached"
        except CannotAnalyze as exc:
            result["reason"] = exc.reason
        except Exception:
            result["reason"] = "analysis_failed"
        finally:
            await file.close()
            body.clear()
            if worker is not None:
                app.state.idle_workers.append(worker)
                app.state.semaphore.release()
            logger.info(json.dumps({"request_id": request.state.request_id, "duration": result.get("duration"), "outcome": result["outcome_code"],
                                    "latency": time.perf_counter() - started, "bundle_id": app.state.bundle.bundle_id}))
        return result

    @app.post("/api/donations")
    async def donate(request: Request, data: DonationRequest):
        if not settings.donations_enabled:
            return JSONResponse(response_data(request, "donations_disabled"), status_code=403)
        if not data.confirm:
            return JSONResponse(response_data(request, "donation_consent_required"), status_code=422)
        if data.consent_version != CONSENT_VERSION:
            return JSONResponse(response_data(request, "consent_version_changed"), status_code=409)
        if not data.eligibility_confirmed:
            return JSONResponse(response_data(request, "donation_eligibility_required"), status_code=422)
        try:
            code = await asyncio.to_thread(app.state.donations.donate, data.token)
        except KeyError:
            return JSONResponse(response_data(request, "expired_or_invalid_token"), status_code=404)
        except CapacityReached:
            return JSONResponse(response_data(request, "donation_capacity_reached"), status_code=507)
        log_storage("donate")
        return {"withdrawal_code": code, "consent_version": CONSENT_VERSION, "status": "quarantined" if settings.donation_store_audio else "consent_recorded"}

    @app.delete("/api/donations/{code}")
    async def withdraw(request: Request, code: str):
        if len(code) > 64:
            return JSONResponse(response_data(request, "withdrawal_not_found"), status_code=404)
        try:
            await asyncio.to_thread(app.state.donations.withdraw, code)
        except KeyError:
            return JSONResponse(response_data(request, "withdrawal_not_found"), status_code=404)
        log_storage("withdraw")
        return {"status": "withdrawn"}

    @app.delete("/api/donation-tokens/{token}")
    async def discard_token(token: str):
        app.state.donations.discard(token)
        return {"status": "discarded"}

    static = settings.root / "static"
    if static.exists():
        app.mount("/static", StaticFiles(directory=static), name="static")
        @app.api_route("/", methods=["GET", "HEAD"])
        async def index():
            return FileResponse(static / "index.html")

        @app.api_route("/privacy", methods=["GET", "HEAD"])
        async def privacy():
            return FileResponse(static / "privacy.html")
    return app


app = create_app()
