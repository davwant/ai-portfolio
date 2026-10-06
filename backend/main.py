from fastapi import FastAPI, UploadFile, File, HTTPException
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY nahi mili")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"


# ============================================================
# STORAGE
# ============================================================

# Permanent documents about Lakshya
DOCUMENT_DIR = Path("my_documents")
DOCUMENT_DIR.mkdir(exist_ok=True)

# Temporary HR job descriptions
TEMP_JD_DIR = Path("temp_jd")
TEMP_JD_DIR.mkdir(exist_ok=True)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


class ChatRequest(BaseModel):
    question: str


resume_schema = Resume.model_json_schema()


# ============================================================
# FILE READING
# ============================================================

def read_pdf(file_path: Path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def read_docx(file_path: Path):

    document = Document(file_path)

    text = ""

    # Read normal paragraphs
    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            text += paragraph.text + "\n"

    # Read tables
    for table in document.tables:

        for row in table.rows:

            for cell in row.cells:

                if cell.text.strip():
                    text += cell.text + "\n"

    return text


def read_document(file_path: Path):

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return read_pdf(file_path)

    elif extension == ".docx":
        return read_docx(file_path)

    else:
        raise ValueError(
            "Only PDF and DOCX files are supported"
        )


# ============================================================
# PERMANENT KNOWLEDGE BASE
# ============================================================

def read_all_documents():

    all_text = ""

    for file_path in DOCUMENT_DIR.iterdir():

        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue

        try:

            text = read_document(file_path)

            all_text += f"""
            
===== {file_path.name} =====

{text}

"""

        except Exception as e:

            print(
                f"Could not read {file_path.name}: {e}"
            )

    return all_text


# ============================================================
# TEMPORARY HR JD
# ============================================================

def read_latest_jd():

    jd_files = [
        file_path
        for file_path in TEMP_JD_DIR.iterdir()
        if file_path.suffix.lower() in [".pdf", ".docx"]
    ]

    if not jd_files:
        return ""

    latest_jd = max(
        jd_files,
        key=lambda file_path: file_path.stat().st_mtime
    )

    return read_document(latest_jd)


# ============================================================
# AI CHAT WITHOUT JD
# ============================================================

def ask_lakshya_ai(
    question: str,
    permanent_knowledge: str,
    jd_text: str = ""
):

    if jd_text.strip():

        jd_section = f"""
        
==============================
OPTIONAL JOB DESCRIPTION
==============================

{jd_text}

The job description above was uploaded temporarily
by the recruiter for this conversation.

"""

    else:

        jd_section = """
        
No job description has been uploaded.

Answer using Lakshya's portfolio information only.

"""

    system_prompt = f"""
You are Lakshya's AI portfolio assistant.

Your job is to help recruiters, HR professionals,
interviewers, and visitors learn about
Lakshya Khandelwal.

You have access to Lakshya's permanent portfolio
information.

==============================
LAKSHYA'S PORTFOLIO
==============================

{permanent_knowledge}

{jd_section}

==============================

Answer the user's question using the information
available above.

IMPORTANT RULES:

1. Do not invent information about Lakshya.

2. Do not assume that Lakshya has a skill,
experience, project, or achievement unless it is
mentioned in his portfolio.

3. If the requested information is not available,
clearly say that it is not available in the portfolio.

4. If a job description is available, you may compare
its requirements with Lakshya's portfolio.

5. When comparing Lakshya with a job description,
clearly distinguish between:
   - What Lakshya actually has
   - What the job requires
   - What is missing or unclear

6. Never assume that Lakshya has a requirement
just because it appears in the job description.

7. Give useful and honest answers.

8. You are an AI assistant representing Lakshya's
portfolio. Do not claim to literally be Lakshya.

9. Keep responses professional but conversational.

User's question:

{question}
"""

    response = client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content

# ============================================================
# OPTIONAL RESUME PARSER
# ============================================================

def parse_resume(resume_text):

    system_prompt = f"""
You are an expert resume parser.

Extract information from the resume based on its meaning,
not only exact section headings.

Return ONLY valid JSON matching this schema:

{resume_schema}

Rules:

1. Do not invent information.
2. If a value is not available, return null.
3. If a list has no information, return an empty list.
4. Include internships inside experiences.
5. Extract skills mentioned throughout the resume.
"""

    user_prompt = f"""
Parse the following resume:

{resume_text}
"""

    response = client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        response_format={
            "type": "json_object"
        }
    )

    raw_output = response.choices[0].message.content

    data = json.loads(raw_output)

    resume = Resume(**data)

    return resume


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Lakshya AI Portfolio",
    description="AI-powered portfolio for Lakshya Khandelwal"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Lakshya AI Portfolio Backend is running 🚀"
    }


# ============================================================
# NORMAL PORTFOLIO CHAT
# ============================================================
@app.post("/chat")
def chat(request: ChatRequest):

    permanent_knowledge = read_all_documents()

    if not permanent_knowledge.strip():

        raise HTTPException(
            status_code=400,
            detail="No permanent portfolio documents uploaded yet"
        )

    # Check whether HR has uploaded a JD
    jd_text = read_latest_jd()

    answer = ask_lakshya_ai(
        question=request.question,
        permanent_knowledge=permanent_knowledge,
        jd_text=jd_text
    )

    return {
        "answer": answer
    }
# ============================================================
# ADMIN: UPLOAD PERMANENT DOCUMENT
# ============================================================

@app.post("/admin/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    allowed_extensions = [".pdf", ".docx"]

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are allowed"
        )

    file_path = DOCUMENT_DIR / Path(file.filename).name

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        text = read_document(file_path)

    except Exception as e:

        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not read document: {str(e)}"
        )

    return {

        "message": "Permanent document uploaded successfully",

        "filename": file.filename,

        "characters": len(text)
    }


# ============================================================
# HR: UPLOAD TEMPORARY JOB DESCRIPTION
# ============================================================

@app.post("/hr/upload-jd")
async def upload_jd(
    file: UploadFile = File(...)
):

    allowed_extensions = [".pdf", ".docx"]

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are allowed"
        )

    file_path = TEMP_JD_DIR / Path(file.filename).name

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    try:

        jd_text = read_document(file_path)

    except Exception as e:

        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not read JD: {str(e)}"
        )

    return {

        "message": "Job description uploaded successfully",

        "filename": file.filename,

        "characters": len(jd_text)
    }
