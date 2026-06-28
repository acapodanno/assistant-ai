from rich import text
import asyncio
import json
from litellm import completion, ChatCompletionMessageToolCall
from loguru import logger
from src.agent.models.agent_response import AgentResponse
from src.agent.config.config_tools import execute_tool, load_tools_definition
class GreenThumbAgent:    
    def __init__(self, model: str = "openai/gpt-4o", max_iterations: int = 5):
        self.model = model
        self.max_iterations = max_iterations
        self.tools = load_tools_definition()
    def run(self, user_message: str, history: list = None) -> str:
        logger.info(f"Starting agent run with user message: {user_message}")
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
                        "→ Se la risposta del RAG non è esatta ma contiene informazioni correlate, riferiscile comunque al cliente "
                        "(es. se chiede della resistenza alla pioggia delle cesoie e il RAG restituisce le caratteristiche generali, "
                        "descrivi quelle caratteristiche e poi spiega che non hai info specifiche sulla pioggia).\n"
                        "→ Usa ESCLUSIVAMENTE la risposta del tool come fonte della tua risposta finale.\n"
                        "→ Nel campo JSON 'sources' dell'AgentResponse, copia l'array 'sources' restituito dal tool RAG.\n"
                        "→ Se l'informazione non è presente, NON inventarla. Dì al cliente che non hai l'informazione e CHIEDI PROATTIVAMENTE: "
                        "'Vuoi che apra un ticket per farti contattare dal servizio clienti?'. In questo caso imposta 'needs_human: true'.\n\n"
                        "## REGOLE DI ESCALATION E TICKET (OBBLIGATORIE)\n"
                        "Devi SEMPRE invocare 'create_ticket' e impostare 'needs_human: true' se il cliente lo richiede esplicitamente o nei seguenti casi:\n"
                        "1. Il cliente dichiara di NON aver ricevuto un ordine che risulta già CONSEGNATO.\n"
                        "2. Il cliente chiede di POSTICIPARE, MODIFICARE L'INDIRIZZO o BLOCCARE una consegna di un ordine già IN SPEDIZIONE.\n"
                        "   → IMPORTANTE: comunica ESPLICITAMENTE al cliente che NON è possibile modificare spedizioni già partite, "
                        "poi apri un ticket ALTRO affinché il team valuti opzioni alternative.\n"
                        "3. Il cliente chiede un RESO o un RIMBORSO.\n"
                        "4. Il cliente segnala un PRODOTTO DANNEGGIATO o non conforme.\n"
                        "5. Il cliente esprime forte insoddisfazione o si lamenta ripetutamente.\n\n"
                        "## ISTRUZIONI PER CREARE IL TICKET (ORDINE OBBLIGATORIO)\n"
                        "STEP 1 — Se il cliente ha fornito un ID ordine: chiama SEMPRE 'get_order_by_id' PRIMA di creare il ticket.\n"
                        "STEP 2 — Chiama 'create_ticket' popolando:\n"
                        "  - 'customer': oggetto con name, email, phone dall'ordine recuperato.\n"
                        "  - 'order_details': prodotti, totale, indirizzo, corriere, tracking dall'ordine.\n"
                        "  - 'issue_type': MANCATA_CONSEGNA | PRODOTTO_DANNEGGIATO | RESO | RIMBORSO | ALTRO.\n"
                        "  - 'issue_description': descrizione precisa del problema segnalato dal cliente.\n"
                        "STEP 3 — Comunica al cliente il numero ticket e che verrà ricontattato.\n"
                    )
            })
        messages.append({"role": "user", "content": user_message})   
        result = self.__react_human_loop__(messages)
        if result:
            logger.info("Agent run completed successfully.")
            return result
            
        logger.error("Agent failed to process the request in time.")
        return json.dumps({
            "answer": "Spiacente, non sono riuscito a elaborare la richiesta in tempo.",
            "confidence": "low",
            "sources": [],
            "needs_human": True
        })

    def __react_human_loop__(self,messages: list[str]) -> str:
        try:
            for step in range(self.max_iterations):
                response = completion(
                    model=self.model,
                    messages=messages,
                    tools=self.tools
                ) 
                response_message = response.choices[0].message
                if response_message.tool_calls:
                    messages.append(response_message)
                    for tool_call in response_message.tool_calls:
                        messages = self.__call_tool_function_(tool_call, messages)
                else:
                    final_response = completion(
                        model=self.model,
                        messages=messages,
                        response_format=AgentResponse
                    )
                    return final_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Error: {e}"
    
    def __call_tool_function_(self,tool_call: ChatCompletionMessageToolCall, messages:list[str]) -> list[str]:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        logger.info(f"Executing tool: {function_name} with args: {function_args}")
        tool_result = execute_tool(function_name, function_args)
        logger.info(f"Tool {function_name} executed successfully.")
        messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    })
        return messages

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    try:
        agent = GreenThumbAgent()
        user_query = "Quali sono le politiche di reso di GreenThumb?"
        logger.info(f"User: {user_query}")
        result_json = agent.run(user_query)
        logger.info(result_json)
    finally:
        # Chiude esplicitamente l'event loop per evitare il
        # ValueError: Invalid file descriptor: -1 di Python 3.13 + httpx
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.close()