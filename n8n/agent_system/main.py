import os
import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from tools import convert_time, http_request, financial_advice

# Load environment variables
load_dotenv()

# Configure the Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[convert_time, http_request, financial_advice],
)

# Initialize FastAPI app
app = FastAPI()

# Data models for requests
class ChatMessage(BaseModel):
    message: str

class TelegramMessage(BaseModel):
    message: str

class WorkflowPayload(BaseModel):
    data: dict


# AI Agent Logic
def run_agent(prompt: str):
    """
    Runs the AI agent with the given prompt.
    """
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return response.text


# API Endpoints
@app.post("/chat")
async def handle_chat(payload: ChatMessage):
    """
    Handles incoming messages from a generic chat.
    """
    response = run_agent(payload.message)
    return {"response": response}


@app.post("/telegram")
async def handle_telegram(payload: TelegramMessage):
    """
    Handles incoming messages from Telegram.
    """
    if "https://" in payload.message or "http://" in payload.message:
        # If the message is a URL, you could add specific logic here
        # For now, we'll just pass it to the agent
        response = run_agent(f"Analyze this URL: {payload.message}")
    else:
        response = run_agent(payload.message)
    return {"response": response}


@app.post("/workflow")
async def handle_workflow(payload: WorkflowPayload):
    """
    Handles triggers from another workflow.
    """
    prompt = f"Process the following data from another workflow: {payload.data}"
    response = run_agent(prompt)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
