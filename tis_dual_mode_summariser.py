# tis_dual_mode_summariser.py
# Transformate Intelligence Suite (Dual Mode + Chatbase Integration + Session Isolation + Invisible Context Injection)

import os
import re
import html
import shutil
import requests
from typing import List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from PyPDF2 import PdfReader
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from PIL import Image as PILImage

# === CONFIGURATION ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

CHATBASE_API_KEY = os.getenv("CHATBASE_API_KEY")  # optional but recommended
CHATBASE_AGENT_ID = os.getenv("CHATBASE_AGENT_ID", "eJ86MiNhODp3671KJoMTN")  # your current ID

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"

# Pricing constants (unchanged)
USD_PER_1K_PROMPT = 0.00015
USD_PER_1K_COMPLETION = 0.00060
CHF_CONVERSION_RATE = 0.80

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input_reports")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "summaries")
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LOGO_PATH = os.path.join(BASE_DIR, "Transformate Logo.png")

DEMO_FILES = [
    "FINMA rs 2011 02 20200101 - Capital buffer and capital planning - banks.pdf",
    "FINMA rs 2023 01 20221207 - Operational risks and resilience - banks.pdf",
    "FINMA rs 2018 02 - Duty to report securities transactions.pdf",
]

# ---------- Helpers ----------

def get_session_folder(session_id: Optional[str]) -> str:
    """Return a safe subfolder for a given session."""
    safe_id = re.sub(r"[^\w\-]", "_", session_id or "default")
    folder = os.path.join(OUTPUT_FOLDER, safe_id)
    os.makedirs(folder, exist_ok=True)
    return folder

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def summarise_text(text: str, filename: str):
    """Generate executive summary and return summary + token/cost stats."""
    prompt = f"""
You are an expert financial analyst.
Summarise the following document into a concise executive summary (max 400 words).
Include:
- Main purpose or regulatory intent
- Key operational or compliance impacts
- 3–5 recommended next steps for an executive team.

Document: {text[:12000]}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    summary = response.choices[0].message.content.strip()
    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
    usd_cost = (prompt_tokens / 1000) * USD_PER_1K_PROMPT + (completion_tokens / 1000) * USD_PER_1K_COMPLETION
    chf_cost = usd_cost * CHF_CONVERSION_RATE

    return summary, prompt_tokens, completion_tokens, total_tokens, usd_cost, chf_cost

# === Chatbase: push summary as "document" (optional) ===
def push_to_chatbase(title: str, summary_text: str) -> bool:
    """Send generated summary text to Chatbase as a document (optional)."""
    if not CHATBASE_API_KEY:
        print("⚠️ CHATBASE_API_KEY not configured — skipping Chatbase upload.")
        return False
    url = f"https://www.chatbase.co/api/v1/chatbots/{CHATBASE_AGENT_ID}/documents"
    headers = {
        "Authorization": f"Bearer {CHATBASE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "name": title,
        "type": "text",
        "content": summary_text[:18000],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        if r.status_code == 200:
            print(f"✅ Summary '{title}' uploaded to Chatbase.")
            return True
        print(f"⚠️ Chatbase upload failed [{r.status_code}]: {r.text}")
        return False
    except Exception as e:
        print(f"❌ Chatbase upload error '{title}': {e}")
        return False

# ---------- DOCX and PDF GENERATION (unchanged visual output) ----------

def save_to_pdf(
    summary_text: str,
    filename: str,
    source_document: Optional[str] = None,
    title: str = "Executive Summary",
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    usd_cost: Optional[float] = None,
    chf_cost: Optional[float] = None,
):
    """Save summary text to a PDF with a left-aligned logo and Source Document metadata."""
    src_name = os.path.basename(source_document) if source_document else ""
    first_line = next((l.strip() for l in summary_text.split("\n") if l.strip()), "")
    clean_first = first_line.replace("**", "").strip("* ").strip()
    m = re.search(r"Executive Summary\s*:\s*(.*)", clean_first, flags=re.I)
    exec_title = m.group(1).strip() if m and m.group(1).strip() else clean_first
    if not exec_title:
        exec_title = os.path.splitext(src_name)[0]
    dt = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%B %d %Y, %H:%M CET")

    summary_text = re.sub(r"^(\*\*)?\s*Executive Summary\s*:?.*\n?", "", summary_text, flags=re.I)

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    cover_title = ParagraphStyle("CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                                 fontSize=24, leading=28, spaceAfter=10, alignment=0)
    cover_label = ParagraphStyle("CoverLabel", parent=styles["BodyText"], fontName="Helvetica-Bold",
                                 fontSize=11, leading=14, spaceAfter=2, alignment=0)
    cover_text = ParagraphStyle("CoverText", parent=styles["BodyText"], fontName="Helvetica",
                                fontSize=11, leading=14, spaceAfter=8, alignment=0)
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14, leading=18, spaceAfter=10, spaceBefore=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=11, leading=15, spaceAfter=10)
    list_item = ParagraphStyle("ListItem", parent=styles["BodyText"], fontName="Helvetica",
                               fontSize=11, leading=15, spaceAfter=10)

    def _md_to_html(text: str) -> str:
        t = html.escape(text)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = t.replace("\r\n", "\n").replace("\n", "<br/>")
        return t

    story = []
    if os.path.exists(LOGO_PATH):
        with PILImage.open(LOGO_PATH) as img:
            w, h = img.size
            target_width = 50 * mm
            target_height = target_width * (h / w)
        story.append(Image(LOGO_PATH, width=target_width, height=target_height, hAlign='LEFT'))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Executive Summary", cover_title))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary Title:", cover_label))
    story.append(Paragraph(html.escape(exec_title), cover_text))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated on: {html.escape(dt)}", cover_text))
    story.append(Paragraph(f"Source Document: {html.escape(src_name)}", cover_text))
    story.append(Paragraph("Prepared by: Transformate Consulting | AI Generated Summary", cover_text))

    if all(v is not None for v in [prompt_tokens, completion_tokens, total_tokens, usd_cost, chf_cost]):
        story.append(Spacer(1, 8))
        story.append(Paragraph("AI Processing Summary:", cover_label))
        story.append(Paragraph(
            html.escape(f"Prompt Tokens: {prompt_tokens:,} | Completion Tokens: {completion_tokens:,} | Total: {total_tokens:,}"),
            cover_text
        ))
        story.append(Paragraph(
            html.escape(f"Estimated Cost: USD ${usd_cost:.4f} ≈ CHF {chf_cost:.4f}"),
            cover_text
        ))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Contact:", cover_label))
    story.append(Paragraph("Tony Ursich", cover_text))
    story.append(Paragraph("Chief Information Officer | Transformate Consulting", cover_text))
    story.append(Paragraph("E: tony.ursich@transformate.ch", cover_text))
    story.append(Paragraph("T: +41 76 577 1165", cover_text))
    story.append(Paragraph("W: www.transformate.ch", cover_text))
    story.append(PageBreak())

    story.append(Paragraph(title, heading))
    for line in [l.strip() for l in summary_text.split("\n") if l.strip()]:
        clean_line = line.strip()
        if clean_line.startswith("**") and clean_line.endswith("**"):
            clean_line = clean_line.replace("**", "")
            story.append(Paragraph(f"<b>{html.escape(clean_line)}</b>", heading))
            story.append(Spacer(1, 6))
            continue
        if re.match(r"^[A-Z].+?:$", clean_line) or (
            "Purpose and Regulatory Intent" in clean_line
            or "Key Operational" in clean_line
            or "Recommended Next Steps" in clean_line
        ):
            story.append(Paragraph(f"<b>{html.escape(clean_line)}</b>", heading))
            story.append(Spacer(1, 8))
            continue
        if re.match(r"^\d+\.", clean_line):
            html_line = _md_to_html(clean_line)
            html_line = re.sub(r"</b>:", r":</b>", html_line)
            story.append(Paragraph(html_line, list_item))
            story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(_md_to_html(clean_line), body))
        story.append(Spacer(1, 8))

    doc.build(story)

# ---------- FastAPI App ----------

app = FastAPI(title="Transformate Intelligence Suite (Dual Mode + Chatbase)", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health():
    return {"status": "ok", "service": "Transformate Intelligence Suite + Chatbase"}

@app.get("/files/{path:path}")
async def get_file(path: str):
    safe_path = os.path.normpath(path).lstrip(os.sep)
    full_path = os.path.join(OUTPUT_FOLDER, safe_path)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path, media_type="application/pdf")

def _summarise_and_generate(pdf_path: str, background_tasks: BackgroundTasks, session_folder: str) -> dict:
    """Summarise a single PDF, generate a summary PDF, save a .txt, and (optionally) push to Chatbase."""
    if not pdf_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {os.path.basename(pdf_path)}")

    text = extract_text_from_pdf(pdf_path)
    summary, pt, ct, tt, usd, chf = summarise_text(text, os.path.basename(pdf_path))

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_pdf = os.path.join(session_folder, f"Summary_{base_name}.pdf")
    txt_path = os.path.join(session_folder, f"Summary_{base_name}.txt")

    # Save PDF
    save_to_pdf(
        summary,
        output_pdf,
        source_document=pdf_path,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
        usd_cost=usd,
        chf_cost=chf,
    )

    # Save TXT (for context retrieval)
    clean_title = base_name
    dt = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d %H:%M CET")
    header = f"Executive Summary Title: {clean_title}\nGenerated: {dt}\n"
    body = re.sub(r"^\s*(Executive Summary\s*:?\s*)", "", summary.strip(), flags=re.I)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + body.strip() + "\n")

    # Optional: push to Chatbase as document (non-blocking)
    if CHATBASE_API_KEY:
        background_tasks.add_task(push_to_chatbase, base_name, summary)

    public_url = f"/files/{os.path.relpath(output_pdf, OUTPUT_FOLDER).replace(os.sep, '/')}"
    return {
        "filename": os.path.basename(pdf_path),
        "summary_file": public_url,
        "token_usage": {"prompt": pt, "completion": ct, "total": tt},
        "cost": {"usd": usd, "chf": chf},
        "summary_preview": summary[:500] + ("..." if len(summary) > 500 else ""),
    }

@app.post("/summarise_demo")
async def summarise_demo(background_tasks: BackgroundTasks, session_id: Optional[str] = Query(default=None)):
    session_folder = get_session_folder(session_id)
    missing = [f for f in DEMO_FILES if not os.path.isfile(os.path.join(INPUT_FOLDER, f))]
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing demo files: {missing}")

    results = []
    for name in DEMO_FILES:
        pdf_path = os.path.join(INPUT_FOLDER, name)
        results.append(_summarise_and_generate(pdf_path, background_tasks, session_folder))

    return JSONResponse({"status": "success", "mode": "demo", "results": results})

@app.post("/summarise_upload")
async def summarise_upload(
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = Query(default=None),
    files: List[UploadFile] = File(...),
):
    session_folder = get_session_folder(session_id)

    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF.")
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 PDFs allowed per request.")

    results = []
    for uf in files:
        if not uf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {uf.filename}")

        safe_name = re.sub(r"[^\w\-. ()]", "_", uf.filename)
        temp_path = os.path.join("/tmp", safe_name)
        with open(temp_path, "wb") as out:
            shutil.copyfileobj(uf.file, out)

        results.append(_summarise_and_generate(temp_path, background_tasks, session_folder))

    return JSONResponse({"status": "success", "mode": "upload", "results": results})

# ---------- Invisible Context Injection → Chatbase (with OpenAI fallback) ----------

def _load_recent_context(session_folder: str, max_files: int = 3, max_chars: int = 12000) -> str:
    """Load up to max_files most-recent .txt summaries and concatenate, truncated."""
    if not os.path.isdir(session_folder):
        return ""
    txts = [f for f in os.listdir(session_folder) if f.lower().endswith(".txt")]
    # Sort by filesystem mtime desc
    txts.sort(key=lambda fn: os.path.getmtime(os.path.join(session_folder, fn)), reverse=True)
    chunks = []
    total = 0
    for fname in txts[:max_files]:
        p = os.path.join(session_folder, fname)
        try:
            with open(p, "r", encoding="utf-8") as f:
                t = f.read().strip()
            if not t:
                continue
            # soft cap to avoid enormous single request
            room = max_chars - total
            if room <= 0:
                break
            chunks.append(t[:room])
            total += min(len(t), room)
        except Exception:
            continue
    return "\n\n".join(chunks).strip()

@app.post("/ask_with_context")
async def ask_with_context(payload: dict, session_id: Optional[str] = Query(default=None)):
    """
    Invisible injection:
      1) Load this session's cached summaries (.txt) as context
      2) Prepend context to the user's query
      3) Send a single user message to Chatbase /api/v1/chat
      4) If Chatbase fails, fallback to OpenAI directly
    """
    user_query = (payload or {}).get("query", "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body.")
    session_folder = get_session_folder(session_id)
    context = _load_recent_context(session_folder)

    injected = (
        "You are the Transformate Intelligence Suite assistant. "
        "Use ONLY the following context to answer. If the context is irrelevant or empty, say you don't have enough information.\n\n"
        f"=== CONTEXT START ===\n{context}\n=== CONTEXT END ===\n\n"
        f"User: {user_query}"
    )

    # Try Chatbase first (keeps the same 'voice' as the widget)
    if CHATBASE_API_KEY:
        try:
            url = "https://www.chatbase.co/api/v1/chat"
            headers = {
                "Authorization": f"Bearer {CHATBASE_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "chatbotId": CHATBASE_AGENT_ID,
                "messages": [{"role": "user", "content": injected}],
            }
            r = requests.post(url, headers=headers, json=body, timeout=25)
            if r.ok:
                data = r.json()
                # Chatbase usually returns {"message": "..."} or {"text": "..."} depending on plan/version
                reply = data.get("message") or data.get("text") or data.get("reply") or ""
                if reply:
                    return {"reply": reply.strip(), "source": "chatbase", "used_context": bool(context)}
                # Some responses come as array
                if isinstance(data, dict) and "messages" in data:
                    msgs = data.get("messages") or []
                    if msgs and isinstance(msgs, list):
                        last = msgs[-1]
                        if isinstance(last, dict):
                            content = last.get("content") or last.get("text") or ""
                            if content:
                                return {"reply": content.strip(), "source": "chatbase", "used_context": bool(context)}
            else:
                print(f"⚠️ Chatbase chat failed [{r.status_code}]: {r.text}")
        except Exception as e:
            print(f"❌ Chatbase chat exception: {e}")

    # Fallback: call OpenAI directly so user isn't blocked
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": injected}],
        temperature=0.3,
    )
    fallback = response.choices[0].message.content.strip()
    return {"reply": fallback, "source": "openai-fallback", "used_context": bool(context)}
