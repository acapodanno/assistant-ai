from src.agent.models.agent_response import AgentResponse
from src.agent.memory import get_formatted_history, add_to_history
from dotenv import load_dotenv
load_dotenv()

import json
from src.rag.run_ingestion import run_ingestion
from src.agent import GreenThumbAgent
from src.api.models.chat_request import ChatRequest
from fastapi import FastAPI
from loguru import logger

run_ingestion()
agent = GreenThumbAgent()
app = FastAPI()
@app.get("/health")
def health():
    logger.info("Health check endpoint called.")
    return {"status": "ok"}

@app.post("/chat/{session_id}")
def chat(session_id: str, chat_request: ChatRequest)-> AgentResponse:
    message = chat_request.message
    logger.info(f"Received chat request: {message} for session: {session_id}")
    formatted_history = get_formatted_history(session_id)
    response_json_str = agent.run(user_message=message, history=formatted_history)
    try:
        response_dict = json.loads(response_json_str)
        text_answer = response_dict.get("answer", response_json_str)
    except json.JSONDecodeError:
        response_dict = {"answer": response_json_str, "confidence": "low", "sources": [], "needs_human": True}
        text_answer = response_json_str

    add_to_history(session_id, message, role="user")
    add_to_history(session_id, text_answer, role="assistant")
    
    logger.info("Chat request processed successfully.")
    return response_dict
    
    