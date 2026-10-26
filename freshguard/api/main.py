"""FastAPI application exposing the FreshGuard pipeline and serving the web UI.

Endpoints
  POST /api/query          run a query (FR1, FR16)
  POST /api/feedback       rate / report an answer (FR18)
  GET  /api/history        recent queries (FR17)
  GET  /api/history/{id}   full stored response
  GET  /api/config         read admin config (FR19)
  POST /api/config         update admin config (FR19)
  GET  /api/kb             list KB docs (FR20)
  POST /api/kb             add/update a KB doc (FR20)
  DELETE /api/kb/{doc_id}  remove a KB doc (FR20)
  GET  /api/eval           run the evaluation module (FR21)
  GET  /api/health         liveness
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from freshguard import __version__
from freshguard.config import FRONTEND_DIR
from freshguard.pipeline import FreshGuard
from freshguard.schemas import (
    QueryRequest, QueryResponse, FeedbackRequest, ConfigUpdate, KBDocument,
)
from freshguard import evaluation


def create_app(fg: FreshGuard | None = None) -> FastAPI:
    app = FastAPI(title="FreshGuard", version=__version__,
                  description="Adaptive, freshness-aware RAG with hallucination detection.")
    engine = fg or FreshGuard()
    app.state.fg = engine

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__,
                "docs_indexed": len(engine.kb_list())}

    @app.post("/api/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        return engine.answer(req.query)

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest):
        if not engine.storage.history_exists(req.history_id):
            raise HTTPException(404, "history_id not found")
        fid = engine.storage.add_feedback(req.history_id, req.rating,
                                          req.report_incorrect, req.comment)
        return {"status": "ok", "feedback_id": fid, "summary": engine.storage.feedback_summary()}

    @app.get("/api/history")
    def history(limit: int = 50):
        return {"items": engine.storage.get_history(limit)}

    @app.get("/api/history/{history_id}")
    def history_item(history_id: int):
        item = engine.storage.get_history_item(history_id)
        if item is None:
            raise HTTPException(404, "not found")
        return item

    @app.get("/api/config")
    def get_config():
        return engine.storage.get_settings().to_dict()

    @app.post("/api/config")
    def set_config(update: ConfigUpdate):
        new = engine.storage.set_config(update.model_dump(exclude_none=True))
        engine._refresh_llm()
        return new.to_dict()

    @app.get("/api/kb")
    def kb_list():
        return {"items": engine.kb_list()}

    @app.post("/api/kb")
    def kb_add(doc: KBDocument):
        doc_id = engine.kb_add(doc.model_dump())
        return {"status": "ok", "doc_id": doc_id}

    @app.delete("/api/kb/{doc_id}")
    def kb_delete(doc_id: str):
        ok = engine.kb_remove(doc_id)
        if not ok:
            raise HTTPException(404, "doc_id not found")
        return {"status": "ok", "removed": doc_id}

    @app.get("/api/eval")
    def run_eval():
        return evaluation.run_all(engine)

    # -- static frontend (FR1/FR16) -------------------------------------------
    if FRONTEND_DIR.exists():
        @app.get("/")
        def index():
            return FileResponse(str(FRONTEND_DIR / "index.html"))
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    else:  # pragma: no cover
        @app.get("/")
        def index():
            return JSONResponse({"message": "FreshGuard API running; frontend not found."})

    return app


app = create_app()
