# Resume Analyzer API

An AI-powered FastAPI service that evaluates a candidate's resume (`.pdf`, `.txt`, or raw text) against a target job description and returns a structured compatibility score, matched/missing skills, reasoning, and token usage metrics.

---

## Architecture

```text
Client (Swagger UI / curl)
        │
        ▼
FastAPI Server (Task_2.py)
  ├── Middleware: Generates unique X-Request-ID & logs lifecycle
  ├── Validation: Checks file type (.pdf/.txt), size (≤5MB), & token limit (≤4,000 tokens)
  ├── Text Extractor: Parses PDF pages via pypdf or decodes plain text
  └── LLM Service (with 15s timeout & 2x retry with backoff)
        │
        ▼
Google Gemini API (Structured JSON Output via Pydantic Schema)
        │
        ▼
JSON Response (Score, Matched/Missing Skills, Reasoning, Request ID, Token Cost)
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Day_5_Tasks
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   ```

---

## How to Run & Test

1. **Start the API server:**
   ```bash
   uvicorn Task_2:app --reload
   ```
   Open interactive API docs at: `http://127.0.0`

2. **Run automated unit tests & coverage:**
   ```bash
   python -m pytest Task_3.py --cov=Task_2 -v
   ```

3. **Run golden-set evaluation:**
   ```bash
   python Task_3_Part2.py
   ```

---

## Example `curl` Commands

**Analyze via plain text:**
```bash
curl -X POST "http://127.0.0" \
  -F "job_description=Looking for a Python backend developer with FastAPI, Docker, and SQL experience." \
  -F "resume_text=Jane Doe. Software Engineer skilled in Python, FastAPI, and PostgreSQL."
```

**Analyze via PDF file upload:**
```bash
curl -X POST "http://127.0.0" \
  -F "job_description=Looking for a Python backend developer with FastAPI, Docker, and SQL experience." \
  -F "resume_file=@sample_resume.pdf;type=application/pdf"
```

---

## Limitations & Fairness Note

* **Limitations:**
  * **Scanned PDFs:** Image-only or scanned PDFs without selectable text cannot be parsed by `pypdf` and will return a `422` error.
  * **Context Window Cap:** Resumes exceeding ~4,000 tokens (~16,000 characters) are rejected (`413`) to prevent latency spikes and high token costs.
  * **LLM Variability:** While `temperature` is kept low and structured schemas enforce JSON format, numerical scores may still vary slightly across runs.
* **Fairness & Responsible Use:**
  * This tool is designed strictly as a **decision-support aid** for initial keyword and skill alignment—never as an automated hiring or rejection system.
  * Prompts explicitly instruct the model to evaluate only technical skills and qualifications against the job description, ignoring demographic proxies (such as names, gender, age, location, or university prestige). A human recruiter should always review candidates holistically.
