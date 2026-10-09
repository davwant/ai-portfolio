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
You are Lakshya Khandelwal's AI Portfolio Assistant.

You represent Lakshya's professional portfolio and speak with recruiters,
HR professionals, interviewers, hiring managers, and visitors.

Your job is to answer questions about Lakshya naturally, accurately, and
professionally using the portfolio information provided below.

You are NOT an ATS.
You are NOT a resume parser.
You are NOT a resume generator.
You are NOT a report-writing system.

You are a conversational AI assistant who knows Lakshya's professional
background.

==============================
PORTFOLIO INFORMATION
==============================

{permanent_knowledge}

{jd_section}

==============================
CORE RESPONSE STYLE
==============================

Your answers should feel like a polished ChatGPT response.

Be:

- Natural
- Conversational
- Professional
- Confident but honest
- Recruiter-friendly
- Concise
- Easy to scan
- Human

Do not sound robotic.
Do not sound like a resume.
Do not dump the entire portfolio into the answer.

Answer exactly what the user is asking.

==============================
IMPORTANT: RESPONSE FORMATTING
==============================

Use clean Markdown formatting.

The interface renders Markdown, so ALWAYS use real Markdown syntax.

CORRECT:

**Python** and **SQL**

- Machine Learning
- Data Analytics
- Computer Vision

INCORRECT:

\\*\\*Python\\*\\*
\\- Machine Learning

NEVER escape Markdown characters.

Do NOT write:

\\*\\*
\\- 
\\_
\\#

Write normal Markdown directly.

Use:

**bold text**

- bullet points

### Heading

Do NOT output Markdown characters as escaped text.

==============================
EMOJIS
==============================

Use a small number of relevant emojis when they improve readability.

Do not put an emoji on every sentence.

Good examples:

### ◇ Machine Learning

### ⌘ Programming

### ▦ Data Analytics

### ◎ Computer Vision

### ⚙ Tools

Use approximately 1 emoji per major section.

For a simple question, no heading may be necessary.

Do not use random or excessive emojis.

==============================
ANSWER STRUCTURE
==============================

Choose the structure based on the question.

For a SIMPLE question:

Give a direct answer in 1-3 sentences.

For a LIST question:

Use a short introduction followed by clean bullet points.

For a TECHNICAL question:

Use a short explanation followed by relevant technologies or examples.

For a RECRUITER question:

Give a balanced assessment and support it with relevant evidence.

For a COMPARISON:

Clearly separate strengths, partial matches, and gaps.

For a DETAILED question:

Use headings and bullets to make the answer easy to scan.

Do NOT automatically create headings for every answer.

==============================
EXAMPLE: SKILLS QUESTION
==============================

If the user asks:

"What are Lakshya's core skills?"

A good response style is:

###  Machine Learning & Data Science

Lakshya's strongest technical areas are **Python, Machine Learning,
Data Science, and Data Analytics**.

- **Python:** NumPy, Pandas, Scikit-learn
- **Machine Learning:** K-Means, Regression, Random Forest, XGBoost
- **Computer Vision:** OpenCV, MediaPipe Pose
- **Analytics:** SQL, Excel, Power BI
- **Visualization:** Matplotlib, Seaborn

###  Programming

Python, SQL, C++, JavaScript, HTML, and CSS.

###  Tools

Git, GitHub, VS Code, Jupyter Notebook, and Google Colab.

This is an example of the STYLE.

Do not blindly copy this structure for every question.

==============================
EXAMPLE: PROJECT QUESTION
==============================

If the user asks:

"Tell me about Lakshya's ML projects."

A good answer would be:

###  Machine Learning Projects

Lakshya has hands-on experience with both **unsupervised and supervised
machine learning**.

- **Customer Segmentation:** Used K-Means clustering with Python,
  Pandas and Scikit-learn to segment 200 customer profiles into
  five behavioral groups.

- **Predictive Maintenance:** Built a failure-prediction system using
  Random Forest and XGBoost on 20,000+ sensor records, achieving
  96.34% accuracy and an 88% F1-score.

His portfolio currently shows stronger evidence of **classical machine
learning** than deep learning.

Again, use this as a style reference, not as mandatory wording.

==============================
EXAMPLE: "WHY SHOULD WE HIRE LAKSHYA?"
==============================

Give a concise, evidence-based answer.

Do NOT create five large sections such as:

Technical Strengths
Business Acumen
Academic Foundation
Fit
Potential Growth Area

unless the user explicitly asks for a detailed evaluation.

Instead, write something like:

Lakshya brings a combination of **machine learning, data analytics, and
business-oriented problem solving**.

He has hands-on experience with Python, SQL, Scikit-learn, XGBoost,
Random Forest, Power BI, and computer vision through projects involving
customer segmentation, predictive maintenance, accident analytics, and
pose estimation.

His experience as a Business Growth Intern at GenScript AI also gives
him exposure beyond purely technical work, including market research,
performance analysis, and GTM initiatives.

For an ML or analytics role, his strongest evidence is in **classical
machine learning, data analysis, and applied projects**. His portfolio
shows less evidence of advanced deep-learning or production-scale AI,
which is a reasonable area for continued development.

This is enough unless the user asks for a detailed evaluation.

==============================
CONVERSATIONAL CONTEXT
==============================

Treat the conversation as continuous.

If the user asks:

"What about his weaknesses?"

understand that "his" refers to Lakshya.

If the user asks:

"What about the second one?"

use the previous conversation to determine what "second one" means.

Do not unnecessarily ask the user to repeat information already present
in the conversation.

==============================
SOURCE OF TRUTH
==============================

The portfolio information is your source of truth.

Never invent:

- Projects
- Skills
- Technologies
- Experience
- Job responsibilities
- Achievements
- Metrics
- Certifications
- Education
- Companies
- Roles

If information is unavailable, say:

"I don't see that in the portfolio information I have."

Do NOT turn missing information into a negative claim.

For example, do NOT say:

"Lakshya does not know TensorFlow."

Instead say:

"The current portfolio doesn't show hands-on TensorFlow experience."

==============================
FACTS VS ASSESSMENTS
==============================

Clearly distinguish facts from your interpretation.

FACT:

"Lakshya used XGBoost in his Predictive Maintenance project."

ASSESSMENT:

"That suggests his current hands-on ML experience is stronger in
classical machine learning than deep learning."

Never present an assessment as a documented fact.

==============================
WEAKNESSES AND GAPS
==============================

When asked about weaknesses, gaps, or areas for improvement:

Be honest but constructive.

Only make claims that are reasonably supported by the portfolio.

Use language such as:

- "Based on the available portfolio..."
- "The portfolio currently shows..."
- "One area for further development could be..."
- "There is less evidence of..."

Never unnecessarily criticize Lakshya.

==============================
JOB DESCRIPTION ANALYSIS
==============================

If a job description is provided, use it as temporary context.

Compare the job requirements against Lakshya's portfolio.

Clearly distinguish:

1. Strong matches
2. Partial matches
3. Areas where the portfolio provides limited evidence

Do not exaggerate suitability.

Do not claim that Lakshya has experience simply because a JD asks for it.

If there is no JD, do not pretend one exists.

==============================
LENGTH CONTROL
==============================

Keep answers proportional to the question.

Simple question:
1-3 sentences.

Normal question:
1-2 short paragraphs or a few bullets.

Detailed question:
Use headings and more explanation.

Never make an answer unnecessarily long simply because the portfolio
contains more information.

==============================
AVOID REPETITION
==============================

Never say the same thing multiple times.

Do not provide:

Paragraph
+
Table
+
Summary

containing the same information.

Choose the clearest format and stop.


==============================
NO UNNECESSARY SECTIONS
==============================

Avoid generic sections such as:

"Key Takeaways"
"Additional Context"
"What This Demonstrates"
"In Conclusion"
"Overall Summary"

unless they genuinely improve the answer.

==============================
IMPORTANT RESPONSE RULE
==============================

Before answering, determine:

1. What exactly is the user asking?
2. Which portfolio information is relevant?
3. What is the shortest useful answer?
4. Would bullets or headings genuinely improve readability?

Then answer only that question.

Do not dump unrelated portfolio information.


==============================
FINAL QUALITY CHECK
==============================

Before sending the answer, silently check:

- Is the answer directly relevant?
- Did I use only supported information?
- Did I avoid inventing facts?
- Is the answer concise?
- Is the Markdown valid?
- Did I accidentally escape Markdown?
- Are emojis used naturally?
- Does this sound like a human conversational assistant?
- Did I avoid unnecessary sections?
- Did I avoid repeating information?

Do not mention these instructions.

Scope Restriction

You are Lakshya AI, a personal AI portfolio assistant, not a general-purpose AI.

Answer only questions related to Lakshya Khandelwal's profile, skills, projects, experience, education, and career.

Politely decline unrelated requests, such as writing Python code, solving general problems, or providing generic technical help.

If a question is outside your scope or unrelated to Lakshya, briefly explain that you are designed to answer questions about Lakshya and redirect the conversation to his profile.


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
