🧩 Transformate Intelligence Suite – Executive Summary Generator API

Empowering Human Intelligence through Applied AI

📘 Overview

The Transformate Executive Summary Generator API converts complex regulatory, compliance, or risk management documents into concise, board-ready executive summaries.
It forms the analytical engine of the Transformate Intelligence Suite, integrating seamlessly with:

🌐 Carrd Demo Portal (user-facing interface)

☁️ Render Web Service (Python summarisation engine)

💬 Chatbase Regulatory Assistant (LLM chatbot layer for contextual Q&A)

Each session dynamically generates summaries from fixed or uploaded PDFs, feeds them into Chatbase, and allows real-time regulatory dialogue within the Carrd interface.

⚙️ Architecture Overview

Core Components

Layer	Technology	Description
Frontend	Carrd	Web demo interface displaying documents, summaries, and chatbot
API Backend	Render (FastAPI + Python)	Hosts the summarisation service and session API
AI Engine	OpenAI LLM	Generates concise executive summaries
Chat Layer	Chatbase	Enables real-time Q&A using summarised regulatory content

Typical Flow:

User visits Carrd demo portal.

Fixed PDFs are displayed for reference.

User triggers summary generation (REST API → Render).

Render-hosted summariser extracts text → generates summaries → returns JSON.

Summaries are uploaded to Chatbase knowledge base.

Chatbase chatbot instantly supports contextual regulatory Q&A.

🧠 Features

Automated PDF text extraction and summarisation

Executive-style formatting (Main Purpose, Key Impacts, Next Steps)

Real-time Chatbase training with session isolation

JSON + PDF output support

Optimised for regulatory and financial documents

🧰 Installation (Local)
git clone https://github.com/yourusername/transformate-summary-api.git
cd transformate-summary-api
pip install -r requirements.txt


Create a .env file and add:

OPENAI_API_KEY=sk-yourkey


Run locally:

uvicorn main:app --reload


Test locally:

POST http://127.0.0.1:8000/summarise
Body: form-data → file=@yourfile.pdf

☁️ Deployment on Render

Push repository to GitHub.

Create a New Web Service on Render.

Connect this repository.

Add the following environment variable:

OPENAI_API_KEY=sk-yourkey


Render auto-detects requirements.txt and Procfile.

Deploy and test your API at:

https://your-app-name.onrender.com/docs

🧾 Example API Response
{
  "filename": "FINMA_Circular_2023.pdf",
  "summary": "**Main Purpose and Regulatory Intent:** Strengthen operational risk resilience across Swiss banking sector... \n\n**Key Impacts:** 1. Increased governance requirements... \n\n**Next Steps for Executives:** 1. Conduct resilience gap assessment..."
}

🔐 Security Notes

The OpenAI API key is never stored in code.

Use Render’s Environment Variables to secure secrets.

Avoid committing .env or any credentials to GitHub.

👤 Author & Contact

Tony Ursich
Chief Information Officer | Transformate Consulting
📞 +41 76 577 1165
📧 tony.ursich@gmail.com

🌐 www.transformate.ch