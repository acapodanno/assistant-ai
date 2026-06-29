import chainlit as cl
import uuid
import json
import asyncio
from typing import Dict, Optional
from dotenv import load_dotenv

from src.agent import GreenThumbAgent, get_formatted_history, add_to_history
from src.rag import run_ingestion

load_dotenv()
run_ingestion()
agent = GreenThumbAgent()


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    await cl.Message(content="Benvenuto su GreenThumb! Sono il tuo assistente virtuale.").send()

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    user_query = message.content
    msg = cl.Message(content="")
    await msg.send()
    formatted_history = get_formatted_history(session_id)
    response_json_str = await asyncio.to_thread(agent.run, user_message=user_query, history=formatted_history)
    try:
        response_dict = json.loads(response_json_str)
        text_answer = response_dict.get("answer", response_json_str)
        sources = response_dict.get("sources", [])
    except json.JSONDecodeError:
        text_answer = response_json_str
        sources = []
    display_text = text_answer
    if sources:
        display_text += "\n\n*(Fonti: " + ", ".join(sources) + ")*"
    add_to_history(session_id, user_query, role="user")
    add_to_history(session_id, text_answer, role="assistant")
    msg.content = display_text
    await msg.update()
