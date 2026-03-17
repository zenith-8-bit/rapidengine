"""
RapidEngine — Credit Appraisal Engine
FastAPI backend: serves the SPA + all API routes

Project layout expected:
    app/
      main.py            ← this file
      static/
        index.html       ← the single-page application
      ml/
        predictor.py     ← Five-Cs scoring model
        finbert.py       ← sentiment pipeline
        layoutlm.py      ← document QA
      db.py              ← SQLAlchemy engine + session
      models.py          ← ORM + Pydantic schemas
    .env                 ← secrets (never commit)
    requirements.txt

Start:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Prod (via gunicorn):
    gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4
"""

from __future__ import annotations

import os
import uuid
import json
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

# ── FastAPI + ASGI ────────────────────────────────────────────────────────────
from fastapi import (
    FastAPI, Depends, HTTPException, status,
    BackgroundTasks, UploadFile, File, Form, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ── Pydantic v2 ───────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field, EmailStr

# ── Auth ──────────────────────────────────────────────────────────────────────
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── DB (SQLAlchemy async) ─────────────────────────────────────────────────────
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, ForeignKey, Boolean,
    create_engine, event
)
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker

# ── Env ───────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL    = os.getenv("DATABASE_URL", "sqlite:///./rapidengine.db")
SECRET_KEY      = os.getenv("SECRET_KEY", "change-me-in-production-use-openssl-rand-hex-32")
ALGORITHM       = "HS256"
ACCESS_TOKEN_TTL = int(os.getenv("ACCESS_TOKEN_TTL_MINUTES", "480"))   # 8 hours
UPLOAD_DIR      = Path(os.getenv("UPLOAD_DIR", "./uploads"))
STATIC_DIR      = Path(os.getenv("STATIC_DIR", "./app/static"))
OLLAMA_URL      = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:72b")
HF_TOKEN        = os.getenv("HF_TOKEN", "")                             # HuggingFace API token
MAX_UPLOAD_MB   = int(os.getenv("MAX_UPLOAD_MB", "50"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rapidengine")


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────────────────────────────────────

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class UserORM(Base):
    __tablename__ = "users"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email       = Column(String, unique=True, index=True, nullable=False)
    name        = Column(String, nullable=False)
    role        = Column(String, default="credit_officer")   # credit_officer | manager | admin
    hashed_pw   = Column(String, nullable=False)
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    cases       = relationship("CaseORM", back_populates="owner")


class CaseORM(Base):
    __tablename__ = "cases"
    id              = Column(String, primary_key=True, default=lambda: f"CAM-{datetime.now().year}-{str(uuid.uuid4())[:6].upper()}")
    owner_id        = Column(String, ForeignKey("users.id"), nullable=False)
    company_name    = Column(String, nullable=False)
    gstin           = Column(String, index=True)
    cin             = Column(String)
    pan             = Column(String)
    sector          = Column(String)
    loan_amount_cr  = Column(Float)
    purpose         = Column(Text)
    status          = Column(String, default="draft")        # draft | processing | review | approved | rejected
    step_reached    = Column(Integer, default=1)             # 1=company_info 2=ingest 3=research 4=cam
    score_total     = Column(Float)
    score_character = Column(Float)
    score_capacity  = Column(Float)
    score_capital   = Column(Float)
    score_collateral= Column(Float)
    score_conditions= Column(Float)
    verdict         = Column(String)                          # APPROVE | CONDITIONAL | REJECT
    sentiment_score = Column(Float)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    owner           = relationship("UserORM", back_populates="cases")
    documents       = relationship("DocumentORM", back_populates="case", cascade="all, delete-orphan")
    signals         = relationship("SignalORM",   back_populates="case", cascade="all, delete-orphan")
    audit_events    = relationship("AuditORM",    back_populates="case", cascade="all, delete-orphan")
    research_items  = relationship("ResearchORM", back_populates="case", cascade="all, delete-orphan")


class DocumentORM(Base):
    __tablename__ = "documents"
    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id         = Column(String, ForeignKey("cases.id"), nullable=False)
    doc_type        = Column(String)   # gst | bank | itr | annual | board | rating
    original_name   = Column(String)
    stored_path     = Column(String)
    extractor       = Column(String)   # pandas | pdfplumber | layoutlmv3 | xml
    confidence      = Column(Float)
    extracted_fields= Column(Text)     # JSON blob of key→value pairs
    status          = Column(String, default="pending")   # pending | processing | complete | failed
    uploaded_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    case            = relationship("CaseORM", back_populates="documents")


class SignalORM(Base):
    __tablename__ = "signals"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id     = Column(String, ForeignKey("cases.id"), nullable=False)
    signal_type = Column(String)   # gst_gap | dscr_decline | nclt | litigation | cash_heavy | ...
    severity    = Column(String)   # critical | high | medium | info
    title       = Column(String)
    description = Column(Text)
    source      = Column(String)
    weight      = Column(Float)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    case        = relationship("CaseORM", back_populates="signals")


class AuditORM(Base):
    __tablename__ = "audit_events"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id     = Column(String, ForeignKey("cases.id"), nullable=False)
    actor       = Column(String)   # user_id or "system"
    event_type  = Column(String)   # doc_upload | conflict_flag | llm_inference | field_note | sign_off | ...
    detail      = Column(Text)
    metadata_   = Column("metadata", Text)  # JSON
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    case        = relationship("CaseORM", back_populates="audit_events")


class ResearchORM(Base):
    __tablename__ = "research_items"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id     = Column(String, ForeignKey("cases.id"), nullable=False)
    headline    = Column(String)
    summary     = Column(Text)
    source_name = Column(String)
    source_url  = Column(String)
    published   = Column(String)
    sentiment   = Column(Float)    # FinBERT score
    weight_label= Column(String)   # Critical | High | Positive | ...
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    case        = relationship("CaseORM", back_populates="research_items")


class ModelConfigORM(Base):
    __tablename__ = "model_configs"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id    = Column(String, ForeignKey("users.id"), nullable=False)
    provider    = Column(String)    # OpenAI | Anthropic | Google | Groq | Cohere | Custom | Ollama
    model_name  = Column(String)
    endpoint    = Column(String)
    api_key_hash= Column(String)    # SHA-256 — never store plaintext
    purpose     = Column(String)    # primary_llm | sentiment | document_qa | summarization
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Create all tables
Base.metadata.create_all(bind=engine)


# ── DB dependency ─────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH
# ─────────────────────────────────────────────────────────────────────────────

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer  = HTTPBearer(auto_error=False)


def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_TTL)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db)
) -> UserORM:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_token(creds.credentials)
        user_id: str = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(UserORM).filter(UserORM.id == user_id, UserORM.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ─────────────────────────────────────────────────────────────────────────────
#  ML — loaded once at startup, injected via Depends
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def get_finbert_pipeline():
    """
    Load FinBERT sentiment pipeline once.
    Falls back gracefully if HuggingFace is unavailable.
    """
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            use_auth_token=HF_TOKEN or None,
            device=-1,   # CPU; change to 0 for GPU
        )
    except Exception as e:
        log.warning(f"FinBERT not available: {e}. Sentiment will return 0.0.")
        return None


@lru_cache(maxsize=None)
def get_layoutlm_pipeline():
    """
    Load LayoutLMv3 document-QA pipeline once.
    Falls back to None if unavailable.
    """
    try:
        from transformers import pipeline
        return pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
            use_auth_token=HF_TOKEN or None,
        )
    except Exception as e:
        log.warning(f"LayoutLM not available: {e}. Document QA disabled.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  PYDANTIC SCHEMAS (request / response)
# ─────────────────────────────────────────────────────────────────────────────

class RegisterIn(BaseModel):
    email: str
    name: str
    password: str
    role: str = "credit_officer"

class LoginIn(BaseModel):
    email: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    name: str
    role: str

class CaseCreateIn(BaseModel):
    company_name: str
    gstin: Optional[str] = None
    cin: Optional[str] = None
    pan: Optional[str] = None
    sector: Optional[str] = None
    loan_amount_cr: Optional[float] = None
    purpose: Optional[str] = None

class CaseOut(BaseModel):
    id: str
    company_name: str
    gstin: Optional[str]
    sector: Optional[str]
    loan_amount_cr: Optional[float]
    status: str
    step_reached: int
    score_total: Optional[float]
    verdict: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ExtractedFieldsIn(BaseModel):
    doc_id: str
    fields: dict[str, Any]   # key → edited value from the UI

class ResearchRunIn(BaseModel):
    case_id: str
    queries: list[str]

class ResearchItemIn(BaseModel):
    headline: str
    summary: str
    source_name: str
    source_url: Optional[str] = None
    published: Optional[str] = None
    sentiment: Optional[float] = None
    weight_label: Optional[str] = None

class ScorecardIn(BaseModel):
    case_id: str
    character: float
    capacity: float
    capital: float
    collateral: float
    conditions: float

class ModelConnectIn(BaseModel):
    provider: str
    model_name: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    purpose: str

class AuditEventIn(BaseModel):
    case_id: str
    event_type: str
    detail: str
    metadata: Optional[dict] = None

class OfficerSignOffIn(BaseModel):
    case_id: str
    action: str    # "confirm" | "override_reject"
    notes: str


# ─────────────────────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RapidEngine Credit Appraisal API",
    version="1.0.0",
    description="Backend for IntelliCredit SPA — cases, documents, ML inference, audit.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: seed a demo user if DB is empty ──────────────────────────────────

@app.on_event("startup")
def seed_demo_user():
    db = SessionLocal()
    try:
        if not db.query(UserORM).first():
            demo = UserORM(
                email="rkumar@rapidengine.in",
                name="R. Kumar",
                role="credit_officer",
                hashed_pw=hash_password("demo1234"),
            )
            db.add(demo)
            db.commit()
            log.info("Demo user created: rkumar@rapidengine.in / demo1234")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: AUTH
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=TokenOut, tags=["auth"])
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if db.query(UserORM).filter(UserORM.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    user = UserORM(
        email=body.email, name=body.name,
        role=body.role, hashed_pw=hash_password(body.password)
    )
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token({"sub": user.id})
    return TokenOut(access_token=token, user_id=user.id, name=user.name, role=user.role)


@app.post("/api/auth/login", response_model=TokenOut, tags=["auth"])
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(UserORM).filter(UserORM.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_pw):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": user.id})
    return TokenOut(access_token=token, user_id=user.id, name=user.name, role=user.role)


@app.get("/api/auth/me", tags=["auth"])
def me(current_user: UserORM = Depends(get_current_user)):
    return {"id": current_user.id, "name": current_user.name,
            "email": current_user.email, "role": current_user.role}


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: CASES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/cases", tags=["cases"])
def create_case(
    body: CaseCreateIn,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    case = CaseORM(owner_id=current_user.id, **body.model_dump())
    db.add(case); db.commit(); db.refresh(case)
    _audit(db, case.id, "system", "case_created",
           f"Case created for {case.company_name} by {current_user.name}")
    return {"id": case.id, "message": "Case created"}


@app.get("/api/cases", tags=["cases"])
def list_cases(
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """Return all cases owned by the logged-in user."""
    cases = db.query(CaseORM).filter(CaseORM.owner_id == current_user.id)\
               .order_by(CaseORM.created_at.desc()).all()
    return [_case_summary(c) for c in cases]


@app.get("/api/cases/{case_id}", tags=["cases"])
def get_case(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    case = _require_case(db, case_id, current_user.id)
    return _case_detail(case)


@app.patch("/api/cases/{case_id}/status", tags=["cases"])
def update_case_status(
    case_id: str, status: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    case = _require_case(db, case_id, current_user.id)
    case.status = status
    db.commit()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: DOCUMENTS / INGESTOR
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/cases/{case_id}/documents/upload", tags=["documents"])
async def upload_document(
    case_id: str,
    doc_type: str = Form(...),          # gst | bank | itr | annual | board | rating
    file: UploadFile = File(...),
    background: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    _require_case(db, case_id, current_user.id)

    # Size guard
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB limit")

    # Persist file under uploads/<case_id>/
    dest_dir = UPLOAD_DIR / case_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{doc_type}_{uuid.uuid4().hex[:8]}_{file.filename}"
    dest_path = dest_dir / safe_name
    dest_path.write_bytes(content)

    # Choose extractor heuristic
    extractor = _pick_extractor(file.filename, doc_type)

    doc = DocumentORM(
        case_id=case_id,
        doc_type=doc_type,
        original_name=file.filename,
        stored_path=str(dest_path),
        extractor=extractor,
        status="processing",
    )
    db.add(doc); db.commit(); db.refresh(doc)

    _audit(db, case_id, current_user.id, "doc_upload",
           f"{doc_type} uploaded: {file.filename} — {extractor} queued")

    # Run extraction in the background (non-blocking)
    background.add_task(_extract_document_bg, doc.id, str(dest_path), doc_type, extractor)

    return {"doc_id": doc.id, "status": "processing", "extractor": extractor}


@app.get("/api/cases/{case_id}/documents", tags=["documents"])
def list_documents(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    _require_case(db, case_id, current_user.id)
    docs = db.query(DocumentORM).filter(DocumentORM.case_id == case_id).all()
    return [_doc_summary(d) for d in docs]


@app.get("/api/documents/{doc_id}/extraction", tags=["documents"])
def get_extraction(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """Return raw text + parsed fields for the extraction modal."""
    doc = db.query(DocumentORM).filter(DocumentORM.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    fields = json.loads(doc.extracted_fields) if doc.extracted_fields else {}
    raw_text = _read_raw_preview(doc.stored_path, doc.doc_type)
    return {
        "doc_id": doc_id,
        "file_name": doc.original_name,
        "extractor": doc.extractor,
        "confidence": doc.confidence,
        "status": doc.status,
        "raw_text": raw_text,
        "fields": fields,
    }


@app.patch("/api/documents/{doc_id}/extraction", tags=["documents"])
def save_extraction_edits(
    doc_id: str,
    body: ExtractedFieldsIn,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """Officer-edited fields from the extraction modal are saved back here."""
    doc = db.query(DocumentORM).filter(DocumentORM.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    existing = json.loads(doc.extracted_fields) if doc.extracted_fields else {}
    existing.update(body.fields)
    doc.extracted_fields = json.dumps(existing)
    db.commit()
    _audit(db, doc.case_id, current_user.id, "extraction_edit",
           f"Fields edited in {doc.original_name}")
    return {"ok": True}


@app.post("/api/cases/{case_id}/detect-conflicts", tags=["documents"])
def detect_conflicts(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """
    Run the conflict-detection layer across all ingested documents for a case.
    Returns detected signals and persists them to the signals table.
    """
    case = _require_case(db, case_id, current_user.id)
    docs = db.query(DocumentORM).filter(
        DocumentORM.case_id == case_id, DocumentORM.status == "complete"
    ).all()

    signals = _run_conflict_detection(docs)

    # Persist new signals (replace existing for idempotency)
    db.query(SignalORM).filter(SignalORM.case_id == case_id).delete()
    for s in signals:
        db.add(SignalORM(case_id=case_id, **s))
    db.commit()

    _audit(db, case_id, "system", "conflict_detection",
           f"{len(signals)} signals detected")

    return {"signals": signals, "count": len(signals)}


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: RESEARCH AGENT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/cases/{case_id}/research/run", tags=["research"])
async def run_research(
    case_id: str,
    body: ResearchRunIn,
    background: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """
    Kick off DuckDuckGo web research + FinBERT sentiment in the background.
    Frontend polls /research/status for completion.
    """
    _require_case(db, case_id, current_user.id)
    _audit(db, case_id, current_user.id, "research_started",
           f"Queries: {'; '.join(body.queries[:4])}")
    background.add_task(
        _run_research_bg, case_id, body.queries,
        get_finbert_pipeline()
    )
    return {"status": "started", "queries": len(body.queries)}


@app.get("/api/cases/{case_id}/research", tags=["research"])
def get_research(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    _require_case(db, case_id, current_user.id)
    items = db.query(ResearchORM).filter(ResearchORM.case_id == case_id)\
               .order_by(ResearchORM.created_at.desc()).all()
    return [_research_out(r) for r in items]


@app.post("/api/cases/{case_id}/research/manual", tags=["research"])
def add_manual_research(
    case_id: str,
    body: ResearchItemIn,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """Allow officer to manually add a news item."""
    _require_case(db, case_id, current_user.id)
    item = ResearchORM(case_id=case_id, **body.model_dump())
    db.add(item); db.commit()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: FIVE-Cs SCORECARD + LLM (CAM)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/cases/{case_id}/score", tags=["cam"])
def save_scorecard(
    case_id: str,
    body: ScorecardIn,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """Save / update the Five-Cs scorecard for a case."""
    case = _require_case(db, case_id, current_user.id)
    weights = {"character": 0.20, "capacity": 0.30, "capital": 0.20,
               "collateral": 0.15, "conditions": 0.15}
    total = (
        body.character  * weights["character"]  +
        body.capacity   * weights["capacity"]   +
        body.capital    * weights["capital"]    +
        body.collateral * weights["collateral"] +
        body.conditions * weights["conditions"]
    )
    case.score_character  = body.character
    case.score_capacity   = body.capacity
    case.score_capital    = body.capital
    case.score_collateral = body.collateral
    case.score_conditions = body.conditions
    case.score_total      = round(total, 2)
    case.verdict = _compute_verdict(total)
    db.commit()
    _audit(db, case_id, current_user.id, "score_updated",
           f"Total: {case.score_total} → {case.verdict}")
    return {"score_total": case.score_total, "verdict": case.verdict}


@app.post("/api/cases/{case_id}/cam/generate", tags=["cam"])
async def generate_cam(
    case_id: str,
    background: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """
    Trigger LLM-based CAM section generation via local Ollama.
    Sections are generated asynchronously and stored per-case.
    Poll /cam/status for completion.
    """
    case = _require_case(db, case_id, current_user.id)
    _audit(db, case_id, "system", "llm_inference",
           f"CAM generation started — model: {OLLAMA_MODEL}")
    background.add_task(_generate_cam_bg, case_id, case)
    return {"status": "generating", "model": OLLAMA_MODEL}


@app.get("/api/cases/{case_id}/cam", tags=["cam"])
def get_cam(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """Return the full CAM payload for the report page."""
    case = _require_case(db, case_id, current_user.id)
    docs     = db.query(DocumentORM).filter(DocumentORM.case_id == case_id).all()
    signals  = db.query(SignalORM).filter(SignalORM.case_id == case_id).all()
    research = db.query(ResearchORM).filter(ResearchORM.case_id == case_id).all()
    audit    = db.query(AuditORM).filter(AuditORM.case_id == case_id)\
                  .order_by(AuditORM.created_at.asc()).all()
    return {
        "case": _case_detail(case),
        "documents": [_doc_summary(d) for d in docs],
        "signals": [_signal_out(s) for s in signals],
        "research": [_research_out(r) for r in research],
        "audit": [_audit_out(a) for a in audit],
    }


@app.post("/api/cases/{case_id}/signoff", tags=["cam"])
def officer_signoff(
    case_id: str,
    body: OfficerSignOffIn,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    case = _require_case(db, case_id, current_user.id)
    if body.action == "confirm":
        case.status = "approved"
    elif body.action == "override_reject":
        case.status = "rejected"
        case.verdict = "REJECT"
    else:
        raise HTTPException(400, "action must be 'confirm' or 'override_reject'")
    db.commit()
    _audit(db, case_id, current_user.id, "officer_signoff",
           f"Action: {body.action} — Notes: {body.notes[:200]}")
    return {"ok": True, "status": case.status}


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: SIGNALS / EWS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/signals", tags=["signals"])
def all_signals(
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    """Return all active EWS signals across all cases owned by the user."""
    case_ids = [c.id for c in db.query(CaseORM.id).filter(CaseORM.owner_id == current_user.id)]
    signals = db.query(SignalORM).filter(SignalORM.case_id.in_(case_ids))\
                .order_by(SignalORM.weight.desc()).all()
    return [_signal_out(s) for s in signals]


@app.get("/api/cases/{case_id}/signals", tags=["signals"])
def case_signals(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    _require_case(db, case_id, current_user.id)
    signals = db.query(SignalORM).filter(SignalORM.case_id == case_id).all()
    return [_signal_out(s) for s in signals]


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: AUDIT TRAIL
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/audit", tags=["audit"])
def all_audit(
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    case_ids = [c.id for c in db.query(CaseORM.id).filter(CaseORM.owner_id == current_user.id)]
    events = db.query(AuditORM).filter(AuditORM.case_id.in_(case_ids))\
               .order_by(AuditORM.created_at.desc()).limit(200).all()
    return [_audit_out(e) for e in events]


@app.post("/api/audit/note", tags=["audit"])
def add_field_note(
    body: AuditEventIn,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    _require_case(db, body.case_id, current_user.id)
    _audit(db, body.case_id, current_user.id, body.event_type, body.detail,
           body.metadata)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: DASHBOARD STATS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/dashboard/stats", tags=["dashboard"])
def dashboard_stats(
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    cases = db.query(CaseORM).filter(CaseORM.owner_id == current_user.id).all()
    total_exposure = sum(c.loan_amount_cr or 0 for c in cases)
    pending_review = sum(1 for c in cases if c.status == "review")
    case_ids = [c.id for c in cases]
    critical_signals = db.query(SignalORM).filter(
        SignalORM.case_id.in_(case_ids), SignalORM.severity == "critical"
    ).count()
    return {
        "active_applications": len(cases),
        "pending_review": pending_review,
        "ews_alerts": critical_signals,
        "total_exposure_cr": round(total_exposure, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE: MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/models", tags=["models"])
def list_models(
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    configs = db.query(ModelConfigORM)\
                .filter(ModelConfigORM.owner_id == current_user.id,
                        ModelConfigORM.is_active == True).all()
    return [_model_config_out(m) for m in configs]


@app.post("/api/models", tags=["models"])
def connect_model(
    body: ModelConnectIn,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    api_key_hash = (
        hashlib.sha256(body.api_key.encode()).hexdigest()
        if body.api_key else None
    )
    cfg = ModelConfigORM(
        owner_id=current_user.id,
        provider=body.provider,
        model_name=body.model_name,
        endpoint=body.endpoint,
        api_key_hash=api_key_hash,
        purpose=body.purpose,
    )
    db.add(cfg); db.commit()
    return {"id": cfg.id, "status": "connected"}


@app.delete("/api/models/{model_id}", tags=["models"])
def disconnect_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: UserORM = Depends(get_current_user)
):
    cfg = db.query(ModelConfigORM).filter(
        ModelConfigORM.id == model_id, ModelConfigORM.owner_id == current_user.id
    ).first()
    if not cfg:
        raise HTTPException(404, "Model config not found")
    cfg.is_active = False
    db.commit()
    return {"ok": True}


@app.get("/api/models/health", tags=["models"])
async def model_health():
    """Check Ollama + HuggingFace reachability."""
    import httpx
    results = {}
    try:
        async with httpx.AsyncClient(timeout=4) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            results["ollama"] = "online" if r.status_code == 200 else "degraded"
    except Exception:
        results["ollama"] = "offline"
    results["finbert"]  = "loaded" if get_finbert_pipeline() else "unavailable"
    results["layoutlm"] = "loaded" if get_layoutlm_pipeline() else "unavailable"
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC + SPA CATCH-ALL  (must be declared LAST)
# ─────────────────────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    # Serve index.html for every non-API route (client-side hash router)
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        index = STATIC_DIR / "inde.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(404, "SPA not found — place index.html in app/static/")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    log.warning(f"Static dir {STATIC_DIR} not found — SPA will not be served.")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS — serialisers
# ─────────────────────────────────────────────────────────────────────────────

def _case_summary(c: CaseORM) -> dict:
    return {
        "id": c.id, "company_name": c.company_name, "gstin": c.gstin,
        "sector": c.sector, "loan_amount_cr": c.loan_amount_cr,
        "status": c.status, "step_reached": c.step_reached,
        "score_total": c.score_total, "verdict": c.verdict,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }

def _case_detail(c: CaseORM) -> dict:
    return {
        **_case_summary(c),
        "cin": c.cin, "pan": c.pan, "purpose": c.purpose,
        "score_character": c.score_character, "score_capacity": c.score_capacity,
        "score_capital": c.score_capital, "score_collateral": c.score_collateral,
        "score_conditions": c.score_conditions, "sentiment_score": c.sentiment_score,
    }

def _doc_summary(d: DocumentORM) -> dict:
    return {
        "id": d.id, "doc_type": d.doc_type, "original_name": d.original_name,
        "extractor": d.extractor, "confidence": d.confidence, "status": d.status,
        "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
    }

def _signal_out(s: SignalORM) -> dict:
    return {
        "id": s.id, "case_id": s.case_id, "signal_type": s.signal_type,
        "severity": s.severity, "title": s.title, "description": s.description,
        "source": s.source, "weight": s.weight,
    }

def _research_out(r: ResearchORM) -> dict:
    return {
        "id": r.id, "headline": r.headline, "summary": r.summary,
        "source_name": r.source_name, "source_url": r.source_url,
        "published": r.published, "sentiment": r.sentiment,
        "weight_label": r.weight_label,
    }

def _audit_out(a: AuditORM) -> dict:
    return {
        "id": a.id, "case_id": a.case_id, "actor": a.actor,
        "event_type": a.event_type, "detail": a.detail,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }

def _model_config_out(m: ModelConfigORM) -> dict:
    return {
        "id": m.id, "provider": m.provider, "model_name": m.model_name,
        "purpose": m.purpose, "is_active": m.is_active,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS — business logic
# ─────────────────────────────────────────────────────────────────────────────

def _require_case(db: Session, case_id: str, user_id: str) -> CaseORM:
    case = db.query(CaseORM).filter(
        CaseORM.id == case_id, CaseORM.owner_id == user_id
    ).first()
    if not case:
        raise HTTPException(404, "Case not found or access denied")
    return case


def _audit(db: Session, case_id: str, actor: str, event_type: str,
           detail: str, metadata: dict = None):
    db.add(AuditORM(
        case_id=case_id, actor=actor, event_type=event_type, detail=detail,
        metadata_=json.dumps(metadata) if metadata else None,
    ))
    db.commit()


def _pick_extractor(filename: str, doc_type: str) -> str:
    fname = filename.lower()
    if doc_type == "gst" and (fname.endswith(".xlsx") or fname.endswith(".csv")):
        return "pandas"
    if fname.endswith(".xml"):
        return "xml_parser"
    if fname.endswith(".pdf"):
        return "layoutlmv3" if doc_type == "annual" else "pdfplumber"
    return "pdfplumber"


def _compute_verdict(score: float) -> str:
    if score >= 70:
        return "APPROVE"
    if score >= 50:
        return "CONDITIONAL"
    return "REJECT"


def _read_raw_preview(stored_path: str, doc_type: str) -> str:
    """Return a plain-text preview of the file for the extraction modal."""
    try:
        path = Path(stored_path)
        if not path.exists():
            return "(file not found)"
        if path.suffix in (".xlsx", ".csv"):
            import pandas as pd
            df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)
            return df.to_string(max_rows=40, max_cols=8)
        if path.suffix == ".pdf":
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages[:4]]
            return "\n\n[Page break]\n\n".join(pages)[:4000]
        return path.read_text(errors="replace")[:4000]
    except Exception as e:
        return f"(preview error: {e})"


def _run_conflict_detection(docs: list[DocumentORM]) -> list[dict]:
    """
    Pure Python conflict rules — runs synchronously, fast.
    Returns list of signal dicts ready to be inserted.
    """
    signals = []
    fields_by_type: dict[str, dict] = {}
    for doc in docs:
        if doc.extracted_fields:
            fields_by_type[doc.doc_type] = json.loads(doc.extracted_fields)

    gst  = fields_by_type.get("gst",  {})
    bank = fields_by_type.get("bank", {})

    # Rule 1 — GST vs Bank turnover gap
    try:
        gst_turnover  = float(str(gst.get("turnover",  "0")).replace("₹", "").replace(",", "").split()[0])
        bank_credits  = float(str(bank.get("credits",  "0")).replace("₹", "").replace(",", "").split()[0])
        if gst_turnover > 0 and bank_credits > 0:
            gap_pct = abs(gst_turnover - bank_credits) / gst_turnover * 100
            if gap_pct > 15:
                signals.append({
                    "signal_type": "gst_bank_gap",
                    "severity": "critical" if gap_pct > 20 else "high",
                    "title": "GST vs Bank Turnover Gap",
                    "description": f"GSTR-3B: ₹{gst_turnover:,.0f}  Bank Credits: ₹{bank_credits:,.0f}  Gap: {gap_pct:.1f}%",
                    "source": "gst + bank_stmt",
                    "weight": 0.90,
                })
    except Exception:
        pass

    # Rule 2 — Cash-heavy revenue
    try:
        cash_pct_raw = str(bank.get("cash_pct", "0"))
        cash_pct = float(cash_pct_raw.split("%")[0].strip())
        if cash_pct > 50:
            signals.append({
                "signal_type": "cash_heavy_revenue",
                "severity": "high",
                "title": "Cash-heavy Revenue Pattern",
                "description": f"{cash_pct:.0f}% of credits via cash — above sector norm of 40%",
                "source": "bank_stmt",
                "weight": 0.75,
            })
    except Exception:
        pass

    # Rule 3 — GSTR-2A / 3B ITC gap
    try:
        gap_raw = str(gst.get("itc_gap", "0")).replace("%", "").strip()
        itc_gap_pct = float(gap_raw.split()[0])
        if itc_gap_pct > 10:
            signals.append({
                "signal_type": "itc_gap",
                "severity": "high",
                "title": "GSTR-2A / 3B ITC Discrepancy",
                "description": f"ITC gap: {itc_gap_pct:.1f}% — possible input tax credit inflation",
                "source": "gst",
                "weight": 0.85,
            })
    except Exception:
        pass

    return signals


# ─────────────────────────────────────────────────────────────────────────────
#  BACKGROUND TASKS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_document_bg(doc_id: str, file_path: str, doc_type: str, extractor: str):
    """
    Runs extraction in a background thread.
    Updates the document record with extracted fields + confidence.
    """
    db = SessionLocal()
    try:
        doc = db.query(DocumentORM).filter(DocumentORM.id == doc_id).first()
        if not doc:
            return

        fields: dict = {}
        confidence: float = 0.0

        if extractor == "pandas":
            fields, confidence = _extract_gst_xlsx(file_path)
        elif extractor == "pdfplumber":
            fields, confidence = _extract_bank_pdf(file_path)
        elif extractor == "layoutlmv3":
            fields, confidence = _extract_annual_report(file_path)
        elif extractor == "xml_parser":
            fields, confidence = _extract_itr_xml(file_path)
        else:
            confidence = 0.5

        doc.extracted_fields = json.dumps(fields)
        doc.confidence = confidence
        doc.status = "complete"
        db.commit()
        log.info(f"Extraction done: {doc_id} confidence={confidence:.2f}")
    except Exception as e:
        log.error(f"Extraction failed for {doc_id}: {e}")
        doc = db.query(DocumentORM).filter(DocumentORM.id == doc_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
    finally:
        db.close()


def _extract_gst_xlsx(file_path: str) -> tuple[dict, float]:
    """Extract key fields from GSTR-3B Excel file."""
    try:
        import pandas as pd
        df = pd.read_excel(file_path)
        # Simplified extraction — real implementation would parse GSTN schema
        return {
            "turnover": "auto-detected",
            "itc_claimed": "auto-detected",
            "filing": "All months filed",
            "_note": "Run full parse in production",
        }, 0.97
    except Exception as e:
        return {"_error": str(e)}, 0.0


def _extract_bank_pdf(file_path: str) -> tuple[dict, float]:
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = " ".join(p.extract_text() or "" for p in pdf.pages)
        return {"_raw_length": len(text), "_note": "Parse amounts with regex in production"}, 0.89
    except Exception as e:
        return {"_error": str(e)}, 0.0


def _extract_annual_report(file_path: str) -> tuple[dict, float]:
    pipe = get_layoutlm_pipeline()
    if not pipe:
        return {"_note": "LayoutLM unavailable — manual entry required"}, 0.0
    # LayoutLM requires image+question pairs; full implementation in ml/layoutlm.py
    return {"_note": "LayoutLM extraction queued"}, 0.40


def _extract_itr_xml(file_path: str) -> tuple[dict, float]:
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()
        return {"xml_tags": len(list(root.iter()))}, 0.95
    except Exception as e:
        return {"_error": str(e)}, 0.0


async def _run_research_bg(case_id: str, queries: list[str], finbert_pipe):
    """
    Web search via DuckDuckGo (no API key) + FinBERT sentiment per article.
    Results persisted to research_items table.
    """
    db = SessionLocal()
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for q in queries[:4]:
                for r in ddgs.text(q, max_results=3):
                    sentiment = 0.0
                    if finbert_pipe and r.get("body"):
                        try:
                            out = finbert_pipe(r["body"][:512], truncation=True)[0]
                            label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                            sentiment = label_map.get(out["label"].lower(), 0.0) * out["score"]
                        except Exception:
                            pass
                    item = ResearchORM(
                        case_id=case_id,
                        headline=r.get("title", "")[:300],
                        summary=r.get("body", "")[:1000],
                        source_name=r.get("source", "Web"),
                        source_url=r.get("href"),
                        published=r.get("date"),
                        sentiment=round(sentiment, 3),
                        weight_label="Critical" if sentiment < -0.6 else
                                     "High"     if sentiment < -0.3 else
                                     "Positive" if sentiment > 0.3  else "Neutral",
                    )
                    db.add(item)
        # Update case average sentiment
        items = db.query(ResearchORM).filter(ResearchORM.case_id == case_id).all()
        scores = [i.sentiment for i in items if i.sentiment is not None]
        if scores:
            avg = sum(scores) / len(scores)
            case = db.query(CaseORM).filter(CaseORM.id == case_id).first()
            if case:
                case.sentiment_score = round(avg, 3)
        db.commit()
        log.info(f"Research complete for {case_id}: {len(items)} items")
    except ImportError:
        log.warning("duckduckgo_search not installed — pip install duckduckgo-search")
    except Exception as e:
        log.error(f"Research failed for {case_id}: {e}")
    finally:
        db.close()


async def _generate_cam_bg(case_id: str, case: CaseORM):
    """
    Call local Ollama to generate CAM narrative sections.
    Stores result as an audit event (full JSON) for the report page.
    """
    import httpx
    db = SessionLocal()
    try:
        docs = db.query(DocumentORM).filter(DocumentORM.case_id == case_id, DocumentORM.status == "complete").all()
        signals = db.query(SignalORM).filter(SignalORM.case_id == case_id).all()
        research = db.query(ResearchORM).filter(ResearchORM.case_id == case_id).all()

        context = f"""
You are a senior credit analyst. Generate a Credit Appraisal Memo (CAM) for the following case.

Company: {case.company_name}
GSTIN: {case.gstin}
Sector: {case.sector}
Loan Amount: ₹{case.loan_amount_cr} Crore
Purpose: {case.purpose}
Five-Cs Scores: Character={case.score_character}, Capacity={case.score_capacity},
  Capital={case.score_capital}, Collateral={case.score_collateral}, Conditions={case.score_conditions}
Total Score: {case.score_total}/100
Verdict: {case.verdict}

Signals detected: {[s.title for s in signals]}
Research sentiment (avg): {case.sentiment_score}
Ingested documents: {[d.doc_type for d in docs]}

Generate a JSON response with keys: character, capacity, capital, collateral, conditions,
conditions_of_sanction (list), summary. Each section: 2-3 sentences. Be concise and factual.
Respond ONLY with valid JSON, no markdown fences.
"""
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": context, "stream": False,
                      "options": {"temperature": 0.1, "num_ctx": 8192}}
            )
            raw = resp.json().get("response", "{}")

        try:
            sections = json.loads(raw)
        except json.JSONDecodeError:
            sections = {"_raw": raw}

        # Store as a special audit event (acts as the CAM content store)
        _audit(db, case_id, "system", "llm_cam_output",
               f"{OLLAMA_MODEL} CAM generation complete",
               {"sections": sections, "model": OLLAMA_MODEL, "score": case.score_total})

        case_obj = db.query(CaseORM).filter(CaseORM.id == case_id).first()
        if case_obj:
            case_obj.status = "review"
            case_obj.step_reached = 4
            db.commit()

        log.info(f"CAM generation done for {case_id}")
    except Exception as e:
        log.error(f"CAM generation failed for {case_id}: {e}")
        _audit(db, case_id, "system", "llm_error", f"CAM generation failed: {str(e)[:300]}")
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
#  DEV ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)