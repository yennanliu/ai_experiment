"""Music Generation App — generates playable music from text prompts using LangChain + OpenAI."""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core import get_llm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Music Generation App")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

llm = get_llm()

generate_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert music composer. Given a description, compose a piece of music in ABC notation.\n\n"
     "Rules:\n"
     "- Output ONLY valid ABC notation, no explanation or markdown\n"
     "- Always include headers: X, T, M, L, K at minimum\n"
     "- Use Q: for tempo\n"
     "- Keep pieces 16-32 bars for reasonable length\n"
     "- Use chords, dynamics, and ornamentation when appropriate\n"
     "- Match the mood, genre, and style described by the user\n\n"
     "Example format:\n"
     "X:1\n"
     "T:Example Tune\n"
     "M:4/4\n"
     "L:1/8\n"
     "Q:1/4=120\n"
     "K:C\n"
     "CDEF GABc|..."),
    ("human", "{prompt}"),
]) | llm | StrOutputParser()


def clean_abc(raw: str) -> str:
    abc = raw.strip()
    if abc.startswith("```"):
        abc = "\n".join(abc.split("\n")[1:])
    if abc.endswith("```"):
        abc = "\n".join(abc.split("\n")[:-1])
    return abc.strip()


def extract_title(abc: str) -> str:
    for line in abc.split("\n"):
        if line.startswith("T:"):
            return line[2:].strip()
    return "Untitled"


def safe_filename(title: str) -> str:
    name = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_').lower()
    return name or "untitled"


class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    abc: str
    title: str
    filename: str

class SaveRequest(BaseModel):
    abc: str
    title: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    abc = clean_abc(generate_chain.invoke({"prompt": req.prompt}))
    title = extract_title(abc)
    filename = f"{safe_filename(title)}_{int(time.time())}.abc"
    # Auto-save
    (OUTPUT_DIR / filename).write_text(abc, encoding="utf-8")
    return GenerateResponse(abc=abc, title=title, filename=filename)


@app.get("/download/{filename}")
def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return {"error": "not found"}
    return FileResponse(path, media_type="text/plain", filename=filename)
