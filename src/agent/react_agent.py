import asyncio
import json
from litellm import completion
from loguru import logger
from src.agent.models.agent_response import AgentResponse
from src.agent.config.config_tools import execute_tool, load_tools_definition

class GreenThumbAgent:    
    def __init__(self, model: str = "openai/gpt-4o", max_iterations: int = 5):
        self.model = model
        self.max_iterations = max_iterations
        self.tools = load_tools_definition()

    def run(self, user_message: str, history: list = None) -> str:
        messages = []
        if history:
            messages.extend(history)
        else:
            messages.append({
                "role": "system", 
                "content": (
                    "Sei l'assistente clienti di GreenThumb Marketplace. Aiuta il cliente usando i tool disponibili.\n\n"
                    "## USO DELLA KNOWLEDGE BASE (RAG)\n"
                    "Per domande su: prodotti, prezzi, guide di giardinaggio, politiche di spedizione, reso, garanzia, pagamenti\n"
                    "→ chiama SEMPRE il tool 'rag_knowledge_base' con la query del cliente PRIMA di rispondere.\n"
                    "→ Se la risposta del RAG non è esatta ma contiene informazioni correlate, riferiscile comunque (es. caratteristiche cesoie).\n"
                    "→ Nel campo JSON 'sources', copia l'array 'sources' del RAG.\n\n"
                    "## REGOLE DI ESCALATION (TICKET)\n"
                    "Devi SEMPRE invocare 'create_ticket' e impostare 'needs_human: true' se:\n"
                    "1. Il cliente non ha ricevuto un ordine già CONSEGNATO.\n"
                    "2. Il cliente vuole cambiare l'indirizzo di un ordine già IN SPEDIZIONE. Comunica che non è possibile farlo, poi apri ticket ALTRO.\n"
                    "3. Il cliente vuole un RESO o RIMBORSO.\n"
                    "4. Il cliente segnala un PRODOTTO DANNEGGIATO.\n\n"
                    "## ISTRUZIONI TICKET (ORDINE OBBLIGATORIO)\n"
                    "STEP 1: Chiama get_order_by_id PRIMA di creare il ticket.\n"
                    "STEP 2: Chiama create_ticket con i dati dell'ordine.\n"
                )
            })
        messages.append({"role": "user", "content": user_message})
        return self.__react_human_loop__(messages)

    def __react_human_loop__(self, messages: list) -> str:
        for _ in range(self.max_iterations):
            response = completion(model=self.model, messages=messages, tools=self.tools)
            msg = response.choices[0].message
            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    res = execute_tool(name, args)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": name, "content": json.dumps(res, ensure_ascii=False)})
            else:
                final = completion(model=self.model, messages=messages, response_format=AgentResponse)
                return final.choices[0].message.content
        return json.dumps({"answer": "Timeout", "sources": [], "needs_human": True})
