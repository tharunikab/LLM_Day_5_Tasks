import asyncio
import io
import json
import logging
import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi_offline import FastAPIOffline
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

# 1. Basic Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("resume-analyzer")

# 2. Load API Key Securely
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in .env file")

client = genai.Client(api_key=API_KEY)

app = FastAPIOffline(
    title="Resume Analyzer API",
    description="Analyzes a resume against a job description with retries, timeouts, and logging.",
)

# Constants for Validation & Limits
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_RESUME_TOKENS = 4000               # Token limit for resume (~16,000 characters)
ALLOWED_CONTENT_TYPES = {"application/pdf", "text/plain"}
LLM_TIMEOUT_SECONDS = 15.0        # Timeout for LLM call
MAX_RETRIES = 2                        # Retry count for transient LLM errors


# 3. Pydantic Schemas
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
    request_id: str
    usage: TokenUsage


# 4. Middleware for Request ID & Logging
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    logger.info(f"[req_id={request_id}] Started {request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(f"[req_id={request_id}] Completed with status {response.status_code}")
    return response


# 5. Helper Functions
def extract_text_from_file(file_bytes: bytes, content_type: str) -> str:
    if content_type == "text/plain":
        return file_bytes.decode("utf-8", errors="ignore").strip()
    elif content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return ""


def estimate_tokens(text: str) -> int:
    # Standard approximation: ~1 token per 4 characters in English text
    return max(1, len(text) // 4)


async def call_llm_with_retry(prompt: str, request_id: str):
    """Calls Gemini with a timeout and retries on timeout, rate limit, or invalid JSON."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[req_id={request_id}] Calling LLM (attempt {attempt}/{MAX_RETRIES})...")

            # Run the SDK call inside asyncio.wait_for to enforce a strict timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AnalysisResult,
                        temperature=0.2,
                    ),
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )

            # Validate that the returned text is valid JSON matching our schema
            parsed_result = AnalysisResult.model_validate_json(response.text)
            return response, parsed_result

        except asyncio.TimeoutError:
            logger.warning(f"[req_id={request_id}] LLM timed out on attempt {attempt}.")
            if attempt == MAX_RETRIES:
                raise HTTPException(
                    status_code=503,
                    detail=f"The AI service took too long to respond (timeout after {LLM_TIMEOUT_SECONDS}s). Please try again shortly. (Request ID: {request_id})",
                )

        except (json.JSONDecodeError, ValidationError):
            logger.warning(f"[req_id={request_id}] LLM returned invalid JSON on attempt {attempt}.")
            if attempt == MAX_RETRIES:
                raise HTTPException(
                    status_code=502,
                    detail=f"The AI service returned an unreadable response format. Please retry. (Request ID: {request_id})",
                )

        except genai_errors.APIError as e:
            logger.warning(f"[req_id={request_id}] Gemini API error (code {e.code}) on attempt {attempt}.")
            # 429 = Rate limit, 503 = Service unavailable
            if e.code in (429, 503):
                if attempt == MAX_RETRIES:
                    raise HTTPException(
                        status_code=503,
                        detail=f"AI service is temporarily busy or rate-limited. Please wait a moment and try again. (Request ID: {request_id})",
                    )
                await asyncio.sleep(1.5 * attempt)  # Brief backoff before retry
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Upstream AI provider error occurred. Please try again later. (Request ID: {request_id})",
                )

        except Exception as e:
            logger.error(f"[req_id={request_id}] Unexpected LLM failure: {type(e).__name__}")
            if attempt == MAX_RETRIES:
                raise HTTPException(
                    status_code=502,
                    detail=f"Unable to complete analysis due to an AI provider issue. (Request ID: {request_id})",
                )


# 6. Endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    request: Request,
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
    request_id = request.state.request_id

    # Failure Case 1: Missing input (400)
    if not resume_file and not resume_text:
        logger.warning(f"[req_id={request_id}] Rejected: No resume provided.")
        raise HTTPException(
            status_code=400,
            detail=f"Please provide either 'resume_file' or 'resume_text'. (Request ID: {request_id})",
        )

    if resume_file:
        # Failure Case 2: Unsupported file type (415)
        if resume_file.content_type not in ALLOWED_CONTENT_TYPES:
            logger.warning(f"[req_id={request_id}] Rejected unsupported type: {resume_file.content_type}")
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{resume_file.content_type}'. Only PDF and TXT are allowed. (Request ID: {request_id})",
            )

        file_bytes = await resume_file.read()

        # Failure Case 3: File size exceeds 5MB (413)
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            logger.warning(f"[req_id={request_id}] Rejected file exceeding 5MB.")
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds the 5MB size limit. (Request ID: {request_id})",
            )

        extracted_resume = extract_text_from_file(file_bytes, resume_file.content_type)
        if not extracted_resume:
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract readable text from the uploaded file. (Request ID: {request_id})",
            )
    else:
        extracted_resume = resume_text.strip()

    # Failure Case 4: Resume exceeds token limit (413)
    estimated_token_count = estimate_tokens(extracted_resume)
    logger.info(f"[req_id={request_id}] Estimated resume tokens: {estimated_token_count}/{MAX_RESUME_TOKENS}")

    if estimated_token_count > MAX_RESUME_TOKENS:
        logger.warning(f"[req_id={request_id}] Rejected resume exceeding token limit ({estimated_token_count} tokens).")
        raise HTTPException(
            status_code=413,
            detail=f"Resume is too long (~{estimated_token_count} tokens). Maximum allowed is {MAX_RESUME_TOKENS} tokens. (Request ID: {request_id})",
        )

    # Call LLM with Timeout & Retry
    prompt = f"Job Description:\n{job_description}\n\nResume:\n{extracted_resume}"
    response, parsed_result = await call_llm_with_retry(prompt, request_id)

    # Calculate Usage & Cost
    usage_meta = response.usage_metadata
    prompt_tokens = usage_meta.prompt_token_count or 0
    completion_tokens = usage_meta.candidates_token_count or 0
    total_tokens = usage_meta.total_token_count or 0

    estimated_cost = (
        (prompt_tokens / 1_000_000) * 0.075
        + (completion_tokens / 1_000_000) * 0.30
    )

    return AnalyzeResponse(
        request_id=request_id,
        **parsed_result.model_dump(),
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
        ),
    )
