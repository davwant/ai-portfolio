from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import shutil
from pathlib import Path
from fastapi.responses import StreamingResponse

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
BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_DIR = BASE_DIR / "my_documents"
DOCUMENT_DIR.mkdir(exist_ok=True)

RESUME_FILE = DOCUMENT_DIR / "resume.txt"
PROFILE_FILE = DOCUMENT_DIR / "profile.txt"
PROJECTS_FILE = DOCUMENT_DIR / "projects.txt"
EXPERIENCE_FILE = DOCUMENT_DIR / "experience.txt"

TEMP_JD_DIR = BASE_DIR / "temp_jd"
TEMP_JD_DIR.mkdir(exist_ok=True)
# ============================================================
# PYDANTIC MODELS
# ============================================================
class ChatRequest(BaseModel):
    question: str


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

    sections = []

    files = [
        ("RESUME", RESUME_FILE),
        ("PERSONAL PROFILE", PROFILE_FILE),
        ("PROJECTS", PROJECTS_FILE),
        ("EXPERIENCE", EXPERIENCE_FILE),
    ]

    for title, file_path in files:

        if file_path.exists():

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if text:

                sections.append(
                    f"""
==============================
{title}
==============================

{text}
"""
                )

    return "\n".join(sections)

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

    stream = client.chat.completions.create(
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
        ],
        stream=True
    )

    def generate():

        for chunk in stream:

            content = chunk.choices[0].delta.content

            if content:
                yield content

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )
# ============================================================
# OPTIONAL RESUME PARSER
# ============================================================




# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Lakshya AI Portfolio",
    description="AI-powered portfolio for Lakshya Khandelwal"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            detail="No permanent portfolio information available"
        )

    jd_text = read_latest_jd()

    return ask_lakshya_ai(
        question=request.question,
        permanent_knowledge=permanent_knowledge,
        jd_text=jd_text
    )
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
