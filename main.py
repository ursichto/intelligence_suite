from fastapi import FastAPI, UploadFile, Form, Response
from batch_report_summariser import summarise_text
from PyPDF2 import PdfReader
import json, os

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

    

# --- Privacy-safe Analytics: HTML Dashboard ---
@app.get("/analytics_overview")
def analytics_overview():
    """Return cumulative analytics with simple daily breakdown if available."""
    store_path = "analytics_store.json"
    if not os.path.exists(store_path):
        return JSONResponse({"message": "No analytics data yet."})

    with open(store_path, "r") as f:
        data = json.load(f)

    # Compute totals
    totals = {k: sum(v.values()) if isinstance(v, dict) else v for k, v in data.items()}
    return JSONResponse({
        "summary": totals,
        "raw_data": data
    })


@app.get("/analytics_data")
def analytics_data():
    """Provide analytics JSON to the dashboard frontend."""
    store_path = "analytics_store.json"
    if not os.path.exists(store_path):
        return {
            "totals": {"sessions": 0, "queries": 0, "uploads": 0, "demo_runs": 0},
            "daily": {}
        }

    with open(store_path, "r") as f:
        data = json.load(f)

    # Handle both formats (your new and old file structures)
    daily = data.get("daily", {})
    totals = data.get("totals", {})

    return {"totals": totals, "daily": daily}




@app.get("/analytics")
def analytics_dashboard():
    # NOTE: Using CDN for Chart.js (no tracking, pure JS bundle)
    return Response(content="""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>TIS Analytics Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root{--bg:#0a2a6c;--panel:#0f3b8f;--ink:#eaf1ff;--muted:#cdd8f5;--accent:#1e90ff;}
    body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
    .wrap{max-width:1100px;margin:24px auto;padding:0 16px}
    header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
    h1{font-size:22px;margin:0}
    .panels{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0 24px}
    .card{background:var(--panel);border-radius:12px;padding:14px}
    .card h3{margin:0 0 6px;font-size:13px;color:var(--muted);font-weight:600}
    .big{font-size:26px;font-weight:800}
    .canvas{background:var(--panel);border-radius:12px;padding:12px;margin-bottom:14px}
    footer{color:var(--muted);font-size:12px;text-align:center;margin-top:10px}
    button.refresh{background:var(--accent);border:none;color:#05223c;border-radius:8px;padding:8px 12px;font-weight:700;cursor:pointer}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Transformate Intelligence Suite — Analytics</h1>
      <div>
        <button class="refresh" id="refresh">Refresh</button>
      </div>
    </header>

    <section class="panels">
      <div class="card"><h3>Total Sessions</h3><div class="big" id="t_sessions">0</div></div>
      <div class="card"><h3>Total Queries</h3><div class="big" id="t_queries">0</div></div>
      <div class="card"><h3>Total Uploads</h3><div class="big" id="t_uploads">0</div></div>
      <div class="card"><h3>Total Demo Runs</h3><div class="big" id="t_demo">0</div></div>
    </section>

    <div class="canvas"><canvas id="dailyLine" height="120"></canvas></div>
    <div class="canvas"><canvas id="dailyStacked" height="140"></canvas></div>

    <footer>Privacy-safe: no cookies, no tracking pixels, no IP storage. Auto-refresh every 60s.</footer>
  </div>

<script>
const fmtInt = (n)=> (n||0).toLocaleString();
let refreshTimer = null;

async function loadData(){
  const res = await fetch('/analytics_data', {cache:'no-store'});
  const data = await res.json();
  renderTotals(data.totals || {});
  renderCharts(data.daily || {});
}

function renderTotals(t){
  document.getElementById('t_sessions').textContent = fmtInt(t.sessions);
  document.getElementById('t_queries').textContent  = fmtInt(t.queries);
  document.getElementById('t_uploads').textContent  = fmtInt(t.uploads);
  document.getElementById('t_demo').textContent     = fmtInt(t.demo_runs);
}

function splitDaily(daily){
  const days = Object.keys(daily).sort(); // YYYY-MM-DD ascending
  const sessions = days.map(d=> daily[d].sessions||0);
  const queries  = days.map(d=> daily[d].queries||0);
  const uploads  = days.map(d=> daily[d].uploads||0);
  const demo     = days.map(d=> daily[d].demo_runs||0);
  return {days, sessions, queries, uploads, demo};
}

let lineChart = null, stackedChart = null;

function renderCharts(daily){
  const {days, sessions, queries, uploads, demo} = splitDaily(daily);

  // Destroy if already exists (prevents duplicate canvas tooltips)
  if (lineChart) lineChart.destroy();
  if (stackedChart) stackedChart.destroy();

  // Multi-series line (trend)
  lineChart = new Chart(
    document.getElementById('dailyLine'),
    {
      type: 'line',
      data: {
        labels: days,
        datasets: [
          {label: 'Sessions', data: sessions, tension: .2},
          {label: 'Queries',  data: queries,  tension: .2},
          {label: 'Uploads',  data: uploads,  tension: .2},
          {label: 'Demo Runs',data: demo,     tension: .2},
        ]
      },
      options: {
        responsive:true,
        maintainAspectRatio:false,
        plugins: {
          legend:{labels:{color:'#eaf1ff'}},
          tooltip:{mode:'index',intersect:false}
        },
        scales:{
          x:{ticks:{color:'#cdd8f5'}, grid:{color:'rgba(255,255,255,0.08)'}},
          y:{ticks:{color:'#cdd8f5'}, grid:{color:'rgba(255,255,255,0.08)'}}
        }
      }
    }
  );

  // Stacked bars (composition per day)
  stackedChart = new Chart(
    document.getElementById('dailyStacked'),
    {
      type:'bar',
      data:{
        labels:days,
        datasets:[
          {label:'Sessions', data:sessions, stack:'stack'},
          {label:'Queries',  data:queries,  stack:'stack'},
          {label:'Uploads',  data:uploads,  stack:'stack'},
          {label:'Demo Runs',data:demo,     stack:'stack'}
        ]
      },
      options:{
        responsive:true,
        maintainAspectRatio:false,
        plugins:{legend:{labels:{color:'#eaf1ff'}}},
        scales:{
          x:{stacked:true, ticks:{color:'#cdd8f5'}, grid:{display:false}},
          y:{stacked:true, ticks:{color:'#cdd8f5'}, grid:{color:'rgba(255,255,255,0.08)'}}
        }
      }
    }
  );
}

document.getElementById('refresh').addEventListener('click', loadData);
loadData();
refreshTimer = setInterval(loadData, 60000); // auto-refresh 60s
</script>
</body>
</html>
""", media_type="text/html")

