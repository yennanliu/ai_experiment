"""Music Generation App — generates playable music from text prompts using LangChain + OpenAI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core import get_llm

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


class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
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
    abc = generate_chain.invoke({"prompt": req.prompt}).strip()
    # Remove markdown fences if LLM wraps it
    if abc.startswith("```"):
        abc = "\n".join(abc.split("\n")[1:])
    if abc.endswith("```"):
        abc = "\n".join(abc.split("\n")[:-1])
    # Extract title from ABC
    title = "Untitled"
    for line in abc.split("\n"):
        if line.startswith("T:"):
            title = line[2:].strip()
            break
    return GenerateResponse(abc=abc, title=title)
