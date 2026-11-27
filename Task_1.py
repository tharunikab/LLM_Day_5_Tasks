import io
import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pypdf import PdfReader

# 1. Load environment variables securely
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in .env file")

client = genai.Client(api_key=API_KEY)

app = FastAPI(
    title="Resume Analyzer API",
    description="Analyzes a resume against a job description and returns a structured match score.",
)

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_CONTENT_TYPES = {"application/pdf", "text/plain"}


# 2. Output Schemas (Matches your Task 1 Expected Output)
class AnalysisResult(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    score: int = Field(description="Match score from 0 to 100", ge=0, le=100)
    matched_skills: list[str] = Field(description="Skills found in both resume and job description")
    missing_skills: list[str] = Field(description="Skills required by job description but missing in resume")
    reasoning: str = Field(description="Brief explanation for the score")


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class AnalyzeResponse(AnalysisResult):
    usage: TokenUsage


# 3. Helper to read PDF or TXT files
def extract_text_from_file(file_bytes: bytes, content_type: str) -> str:
    if content_type == "text/plain":
        return file_bytes.decode("utf-8", errors="ignore")
    elif content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return ""


# 4. Endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    job_description: str = Form(
        ...,
        example="Looking for a Python backend developer with FastAPI, Docker, and SQL experience.",
    ),
    resume_text: Optional[str] = Form(
        None,
        example="Jane Doe. Software Engineer skilled in Python, FastAPI, and PostgreSQL.",
    ),
    resume_file: Optional[UploadFile] = File(None),
):
    # Validation: Must provide either file or text
    if not resume_file and not resume_text:
        raise HTTPException(
            status_code=400,
            detail="You must provide either 'resume_file' or 'resume_text'.",
        )

    # Validation: File type and size
    if resume_file:
        if resume_file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{resume_file.content_type}'. Only PDF and TXT are allowed.",
            )

        file_bytes = await resume_file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum allowed size is 5MB.",
            )

        extracted_resume = extract_text_from_file(file_bytes, resume_file.content_type)
        if not extracted_resume:
            raise HTTPException(
                status_code=422,
                detail="Could not extract readable text from the uploaded file.",
            )
    else:
        extracted_resume = resume_text.strip()

    # Call Gemini AI
    prompt = f"Job Description:\n{job_description}\n\nResume:\n{extracted_resume}"

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResult,
                temperature=0.2,
            ),
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to process request with AI provider.")

    # Calculate tokens & cost
    usage_meta = response.usage_metadata
    prompt_tokens = usage_meta.prompt_token_count or 0
    completion_tokens = usage_meta.candidates_token_count or 0
    total_tokens = usage_meta.total_token_count or 0

    estimated_cost = (
        (prompt_tokens / 1_000_000) * 0.075
        + (completion_tokens / 1_000_000) * 0.30
    )

    parsed_result = AnalysisResult.model_validate_json(response.text)

    return AnalyzeResponse(
        **parsed_result.model_dump(),
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
        ),
    )
