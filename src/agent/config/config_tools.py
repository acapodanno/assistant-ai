import os
import json

from src.agent.tools import get_order_by_id, create_ticket, rag_knowledge_base

def load_tools_definition() -> list:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    tools_path = os.path.join(project_root, "config", "litellm", "tools.json")
    with open(tools_path, "r") as f:
        return json.load(f)
AVAILABLE_TOOLS = {
    "get_order_by_id": get_order_by_id,
    "create_ticket": create_ticket,
    "rag_knowledge_base": rag_knowledge_base,
}

def execute_tool(name: str, arguments: dict) -> dict:
    if name not in AVAILABLE_TOOLS:
        return {"error": f"Tool '{name}' non trovato."}
    
    tool_func = AVAILABLE_TOOLS[name]
    try:
        return tool_func(**arguments)
    except Exception as e:
        return {"error": f"Errore nell'esecuzione del tool '{name}': {str(e)}"}


if __name__ == "__main__":
    print(load_tools_definition())