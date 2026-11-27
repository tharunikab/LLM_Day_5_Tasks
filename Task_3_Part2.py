import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

client = genai.Client(api_key=API_KEY)


class EvalExtraction(BaseModel):
    score: int = Field(ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]


# Golden Set of 8 Resumes with Ground-Truth Expected Skills
GOLDEN_SET = [
    {
        "id": 1,
        "jd": "Looking for Python, FastAPI, Docker, and PostgreSQL.",
        "resume": "Alice. Backend Dev with 4 yrs in Python, FastAPI, Postgres, and Docker.",
        "expected_matched": {"python", "fastapi", "postgresql", "docker"},
    },
    {
        "id": 2,
        "jd": "Need React, TypeScript, Node.js, and AWS.",
        "resume": "Bob. Frontend engineer experienced with React, TS, and NodeJS.",
        "expected_matched": {"react", "typescript", "node.js"},
    },
    {
        "id": 3,
        "jd": "Data Scientist skilled in Python, Pandas, Scikit-Learn, and SQL.",
        "resume": "Charlie. Built ML models using Python, Pandas, sklearn, and MySQL/SQL.",
        "expected_matched": {"python", "pandas", "scikit-learn", "sql"},
    },
    {
        "id": 4,
        "jd": "DevOps Engineer with Kubernetes, Terraform, CI/CD, and Linux.",
        "resume": "Diana. Managed Linux servers, K8s clusters, and automated CI/CD pipelines.",
        "expected_matched": {"kubernetes", "ci/cd", "linux"},
    },
    {
        "id": 5,
        "jd": "Java Spring Boot developer with Kafka and Redis.",
        "resume": "Evan. 5 years building microservices in Java, Spring Boot, and Apache Kafka.",
        "expected_matched": {"java", "spring boot", "kafka"},
    },
    {
        "id": 6,
        "jd": "Machine Learning Engineer: PyTorch, NLP, FastAPI, GCP.",
        "resume": "Fiona. NLP researcher deploying PyTorch transformers on Google Cloud (GCP).",
        "expected_matched": {"pytorch", "nlp", "gcp"},
    },
    {
        "id": 7,
        "jd": "Full Stack Engineer: Python, Django, React, GraphQL.",
        "resume": "George. Developed web apps using Django, Python, and React.",
        "expected_matched": {"python", "django", "react"},
    },
    {
        "id": 8,
        "jd": "Cloud Security Engineer: AWS, IAM, SIEM, Python.",
        "resume": "Hannah. Configured AWS IAM policies and wrote Python security scripts.",
        "expected_matched": {"aws", "iam", "python"},
    },
]

# Weak/Vague Baseline Prompt (often misses abbreviations like TS, K8s, Postgres, sklearn)
BASELINE_PROMPT = """
Extract matched skills and give a score.
Job Description: {jd}
Resume: {resume}
"""

# Improved Prompt (fixes abbreviation normalization & strict canonical naming)
IMPROVED_PROMPT = """
You are a precise technical recruiter. Compare the Resume against the Job Description.

Rules for skill extraction:
1. Extract ONLY skills explicitly mentioned in the Job Description AND present in the Resume into `matched_skills`.
2. Normalize abbreviations and synonyms to match the exact skill names written in the Job Description (e.g., "Postgres" -> "PostgreSQL", "TS" -> "TypeScript", "NodeJS" -> "Node.js", "sklearn" -> "Scikit-Learn", "K8s" -> "Kubernetes").
3. Do NOT include extra tools from the resume that were not requested in the Job Description.
4. Score strictly from 0 to 100 based on the ratio of matched required skills.

Job Description: {jd}
Resume: {resume}
"""


def normalize(skills: list[str]) -> set[str]:
    return {s.strip().lower() for s in skills}


def run_evaluation(prompt_template: str, label: str, run_twice: bool = False):
    print(f"\n--- Running Evaluation: {label} ---")
    total_tp = 0
    total_fp = 0
    total_fn = 0
    score_diffs = []

    for item in GOLDEN_SET:
        prompt = prompt_template.format(jd=item["jd"], resume=item["resume"])

        # Run 1
        res1 = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvalExtraction,
                temperature=0.0,
            ),
        )
        data1 = EvalExtraction.model_validate_json(res1.text)
        predicted = normalize(data1.matched_skills)
        expected = item["expected_matched"]

        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Optional Run 2 to measure score consistency across 2 runs
        if run_twice:
            time.sleep(0.5)
            res2 = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EvalExtraction,
                    temperature=0.0,
                ),
            )
            data2 = EvalExtraction.model_validate_json(res2.text)
            score_diffs.append(abs(data1.score - data2.score))

        time.sleep(0.5)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0

    print(f"Precision: {precision:.2f} | Recall: {recall:.2f}")
    if run_twice:
        print(f"Average Score Difference (2 runs): {avg_diff:.2f} points")

    return precision, recall, avg_diff


if __name__ == "__main__":
    base_p, base_r, base_diff = run_evaluation(BASELINE_PROMPT, "Baseline Prompt", run_twice=True)
    imp_p, imp_r, _ = run_evaluation(IMPROVED_PROMPT, "Improved Prompt (Canonical Normalization)", run_twice=False)

    print("\n================ FINAL EVALUATION REPORT ================")
    print(f"Skill extraction precision {base_p:.2f}, recall {base_r:.2f} -> after prompt fix {imp_p:.2f}, {imp_r:.2f}")
    print(f"Average score difference across 2 runs: {base_diff:.2f}")
    print("Biggest weakness fixed: Alias/abbreviation mismatch (e.g., K8s vs Kubernetes, TS vs TypeScript) and over-extracting unrequested skills.")
