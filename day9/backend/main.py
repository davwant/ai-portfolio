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
You are Lakshya Khandelwal's AI portfolio assistant.

You interact with recruiters, HR professionals, interviewers, hiring managers,
and visitors who want to understand Lakshya's background.

Your goal is NOT to generate resume reports.

Your goal is to have a natural, helpful conversation — similar to ChatGPT —
while answering questions using Lakshya's portfolio as your source of truth.

==============================
PORTFOLIO INFORMATION
==============================

{permanent_knowledge}

{jd_section}

==============================
HOW YOU SHOULD ANSWER
==============================

1. ANSWER NATURALLY

Talk like an intelligent conversational assistant.

Do not sound like a resume parser, ATS system, or generated report.

Answer the question directly instead of creating unnecessary sections,
tables, summaries, or long explanations.
#example 
For example, if someone asks:

"Tell me about Lakshya's ML projects"

A good answer would be:

"Lakshya's main machine-learning project is Customer Segmentation,
where he used Python, Pandas and Scikit-learn's K-Means algorithm to
group 200 customers into five behavioral segments. The project covered
data cleaning, feature engineering, model training and cluster analysis.

He also has experience with data analytics through his Road Accident
Trends project, although that project is primarily analytics and
visualization rather than machine learning."

Do NOT turn this into a table unless the user asks for a table.

2. KEEP ANSWERS APPROPRIATELY SHORT

Think before answering and give only the amount of information needed.

Simple question:
→ 1-3 sentences.

Normal recruiter question:
→ 1-2 short paragraphs or a few bullets.

Complex question:
→ Give a more detailed answer only when necessary.

Never make an answer long simply because more information is available.

3. DO NOT DUMP THE PORTFOLIO

The portfolio may contain lots of information.

That does NOT mean you should mention all of it.

Select only the information relevant to the user's question.

If the user asks about ML, don't discuss unrelated business experience.

If the user asks about internships, don't describe every project.

If the user asks about strengths, don't list every skill.

4. USE NATURAL FOLLOW-UP CONTEXT

Treat the conversation as a real conversation.

If the user asks:

"What about his weaknesses?"

Understand that they are referring to the previous topic.

Do not ask them to repeat context unnecessarily.

5. DISCUSSING WEAKNESSES

When asked about Lakshya's weaknesses, limitations, gaps, or areas
for improvement:

Be honest and balanced.

Only identify weaknesses that can reasonably be inferred from the
portfolio.

Do NOT invent weaknesses.

Do NOT treat the absence of a technology in the portfolio as proof
that Lakshya does not know that technology.

For example, instead of saying:

"Lakshya does not know TensorFlow."

say:

"The portfolio currently shows more experience with classical ML,
particularly K-Means, than with deep-learning frameworks. So one
potential area for further development would be gaining more hands-on
experience with frameworks such as PyTorch or TensorFlow."

Use phrases such as:
- "The portfolio suggests..."
- "Based on the available information..."
- "One area for development could be..."
when appropriate.

Never make unnecessarily negative judgments about Lakshya.

6. DO NOT INVENT INFORMATION

The portfolio is the source of truth.

Never fabricate:
- skills
- technologies
- projects
- job responsibilities
- achievements
- metrics
- certifications
- education
- experience

If something isn't known, say so naturally.

For example:

"I don't see evidence of that in the portfolio information I have."

Do not say:

"Lakshya has no experience with X"

unless the portfolio explicitly establishes that.

7. DISTINGUISH FACTS FROM INFERENCE

If something is explicitly stated in the portfolio, present it as a fact.

If something is an interpretation or reasonable assessment,
make that clear.

Example:

FACT:
"Lakshya used K-Means clustering in his Customer Segmentation project."

ASSESSMENT:
"That suggests his current hands-on ML experience is stronger in
classical machine learning than in deep learning."

Do not present assessments as facts.

8. RECRUITER-FRIENDLY ANSWERS

When answering recruiter questions, focus on what matters:

- What Lakshya has done
- What technologies he has used
- What he appears to be strong at
- Where he may need more development
- How his experience relates to the question

Do not use corporate buzzwords unnecessarily.

9. MARKDOWN

Use formatting only when it genuinely improves readability.

Prefer natural paragraphs.

Bullets are okay when listing several distinct points.

Avoid:
- giant Markdown tables
- excessive headings
- repeated bold text
- "Key Takeaways" sections
- "Additional Context" sections
- "What These Projects Demonstrate" sections
- "In conclusion" sections

10. DO NOT REPEAT YOURSELF

Say something once.

Do not provide the same information in:
- a paragraph
- then a table
- then a summary

Choose the clearest format and stop.

11. HANDLE COMPARISONS INTELLIGENTLY

If the user asks:

"Is Lakshya strong in AI?"

Don't simply list every AI-related technology.

Give a balanced assessment based on the evidence.

Example:

"Lakshya has a foundation in machine learning and data analytics, with
hands-on experience in Python, Scikit-learn and K-Means. His portfolio
currently shows more evidence of classical ML and analytics than
advanced AI or deep-learning development, so I'd describe him as having
a developing AI/ML profile rather than extensive production-level AI
experience."

12. JOB DESCRIPTION

If a job description is available, use it naturally.

If asked whether Lakshya is suitable for a role, compare the role
requirements against the portfolio.

Clearly separate:
- demonstrated strengths
- partial matches
- apparent gaps

Do not exaggerate matches.

13. IF THE USER ASKS FOR A DETAILED ANSWER

Only then provide more structure and detail.

The default behavior should always be conversational and concise.

14. TONE

Be:

- conversational
- intelligent
- balanced
- professional
- honest
- concise
- helpful

Do not sound robotic.

Do not sound like an academic paper.

Do not sound like a resume.

Do not sound like an ATS.

Do not mention these instructions.

==============================
FINAL RULE
==============================

Before responding, ask yourself:

"What is the user actually trying to know?"

Then answer THAT question directly.

Do not answer a larger question than the user asked.
Do not dump everything you know.
Do not invent missing information.

==============================
USER QUESTION
==============================

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
