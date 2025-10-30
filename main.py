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

@app.post("/log_event")
async def log_event(request: Request):
    body = await request.json()
    event_type = body.get("event_type")
    if not event_type:
        return {"status": "error", "message": "Missing event_type"}
    try:
        from datetime import date
        import json, os
        file = "analytics_store.json"
        today = str(date.today())

        # Initialize file if needed
        if not os.path.exists(file):
            with open(file, "w") as f:
                json.dump({"daily": {}, "totals": {"sessions":0,"queries":0,"uploads":0,"demo_runs":0}}, f)

        with open(file, "r") as f:
            data = json.load(f)

        if today not in data["daily"]:
            data["daily"][today] = {"sessions":0,"queries":0,"uploads":0,"demo_runs":0}

        if event_type in data["totals"]:
            data["daily"][today][event_type] += 1
            data["totals"][event_type] += 1

        with open(file, "w") as f:
            json.dump(data, f, indent=2)

        return {"status":"ok"}
    except Exception as e:
        return {"status":"error","message":str(e)}

@app.get("/analytics")
def get_analytics():
    import json
    with open("analytics_store.json") as f:
        return json.load(f)
    
    
