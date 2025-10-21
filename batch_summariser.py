import os
import re
import html
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from openai import OpenAI

# === CONFIGURATION ===
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
OUTPUT_FOLDER = "summaries"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
client = OpenAI(api_key=API_KEY)

USD_PER_1K_PROMPT = 0.00015
USD_PER_1K_COMPLETION = 0.00060
CHF_CONVERSION_RATE = 0.80


# === HELPERS ===
def extract_text_from_pdf(path):
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()


def summarise_text(text, filename):
    """Generate executive summary using GPT model."""
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

    usd_cost = (
        (usage.prompt_tokens / 1000) * USD_PER_1K_PROMPT
        + (usage.completion_tokens / 1000) * USD_PER_1K_COMPLETION
    )
    chf_cost = usd_cost * CHF_CONVERSION_RATE

    return summary, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, usd_cost, chf_cost


def save_to_pdf(summary_text, filename, source_document, cost_data):
    """Save summary text to a PDF with branding and token stats."""
    from reportlab.platypus import HRFlowable
    from reportlab.lib import colors

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14, leading=18, spaceAfter=10)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=11, leading=15, spaceAfter=10)

    dt = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%B %d %Y, %H:%M CET")
    src_name = os.path.basename(source_document)
    usd_cost, chf_cost, total_tokens = cost_data

    story = [
        Paragraph("Executive Summary", heading),
        Spacer(1, 12),
        Paragraph(f"Generated on: {html.escape(dt)}", body),
        Paragraph(f"Source Document: {html.escape(src_name)}", body),
        Paragraph(f"Token Usage: {total_tokens:,} | Cost: USD ${usd_cost:.4f} ≈ CHF {chf_cost:.4f}", body),
        Paragraph("Prepared by: Transformate Consulting | AI Generated Summary", body),
        HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey),
        Spacer(1, 12),
    ]

    for line in summary_text.split("\n"):
        if line.strip():
            story.append(Paragraph(html.escape(line.strip()), body))
            story.append(Spacer(1, 6))

    doc.build(story)


# === FASTAPI APP ===
app = FastAPI(title="Transformate Intelligence Suite", version="1.0")


@app.post("/summarise")
async def generate_summary(file: UploadFile = File(...)):
    """Upload a PDF, generate summary, and return metadata."""
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(temp_path)
    summary, pt, ct, tt, usd, chf = summarise_text(text, file.filename)

    output_pdf = os.path.join(OUTPUT_FOLDER, f"Summary_{os.path.splitext(file.filename)[0]}.pdf")
    save_to_pdf(summary, output_pdf, file.filename, (usd, chf, tt))

    return JSONResponse({
        "status": "success",
        "filename": file.filename,
        "summary_file": output_pdf,
        "token_usage": {"prompt": pt, "completion": ct, "total": tt},
        "cost": {"usd": usd, "chf": chf},
        "summary_preview": summary[:400] + "..."
    })


@app.get("/")
async def healthcheck():
    """Health check endpoint."""
    return {"status": "ok", "service": "Transformate Intelligence Suite"}


# === OPTIONAL: CHATBASE SYNC (for later demo phase) ===
# Add a POST endpoint here to push generated summaries to Chatbase if desired.
