from fastapi import FastAPI, UploadFile, Form
from batch_report_summariser import summarise_text
from PyPDF2 import PdfReader
import os

app = FastAPI()

@app.post("/summarise")
async def summarise_file(file: UploadFile):
    """Receives a PDF file, extracts text, and returns a summary."""
    # Save temp file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    reader = PdfReader(temp_path)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Run your summariser (calls OpenAI)
    summary, *_ = summarise_text(text, file.filename)

    # Cleanup
    os.remove(temp_path)

    return {"filename": file.filename, "summary": summary}
