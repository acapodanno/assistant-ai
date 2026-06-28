from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from typing import Any
from litellm import completion
from pydantic import PrivateAttr
import tiktoken

from loguru import logger

class SummaryBufferHistory(InMemoryChatMessageHistory):
    llm: Any
    max_token_limit: int = 100
    _summary: str = PrivateAttr(default="")
    _enc: Any = PrivateAttr(default=None)

    def __init__(self, llm: Any, max_token_limit: int = 100, **kwargs: Any):
        super().__init__(llm=llm, max_token_limit=max_token_limit, **kwargs)
        object.__setattr__(self, "_enc", tiktoken.encoding_for_model("gpt-3.5-turbo"))
        object.__setattr__(self, "_summary", "")

    def _count_tokens(self) -> int:
        return sum(
            len(self._enc.encode(m.content)) + 4
            for m in self.messages
        )

    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        if self._count_tokens() > self.max_token_limit:
            self._summarize()

    def _summarize(self):
        old_summary = ""
        recent_messages = []
        if not self.messages:
            return
        for m in self.messages:
            if isinstance(m, SystemMessage) and "Riassunto" in m.content:
                old_summary = m.content
            else:
                recent_messages.append(m)

        history_text = "\n".join(
            f"{m.type}:{m.content}" for m in recent_messages
        )
        prompt = f"""Aggiorna il riassunto della conversazione di supporto clienti di GreenThumb Marketplace.        
        Riassunto precedente:
        {old_summary}
        Nuovi messaggi di interazione:
        {history_text}
        
        REGOLE TASSATIVE PER IL NUOVO RIASSUNTO:
        1. Includi l'identità del cliente (Nome, Email o Telefono) se è stata specificata.
        2. Riassumi sinteticamente tutti i dettagli dell'ordine citati (es. ID dell'ordine, se è in spedizione, consegnato o in reso).
        3. Elenca brevemente i prodotti discussi (es. vanga, lavanda, bulbi) e il motivo del contatto (es. richiesta info, ritardo consegna, avvio reso).
        4. Specifica se è stato attivato il tool di escalation o aperto un ticket di supporto.
        5. Formula il riassunto in terza persona in modo fattuale, privo di convenevoli."""
        response = completion(
            model=self.llm,
            messages=[{"role": "user", "content": prompt}]
        )
        object.__setattr__(self, "_summary", response.choices[0].message.content)
        self.clear()
        if self._summary:
            self.messages.append(
                SystemMessage(content=f"Riassunto: {self._summary}")
            )
        logger.info(f"Summary updated: {self._summary}")

# --- Utility per la gestione sessioni (usate in FastAPI) ---
_session_store = {}

def get_session_history(session_id: str) -> SummaryBufferHistory:
    """Recupera o crea la memoria per una data sessione."""
    if session_id not in _session_store:
        _session_store[session_id] = SummaryBufferHistory(llm="gpt-4o-mini", max_token_limit=150)
    return _session_store[session_id]

def get_formatted_history(session_id: str) -> list[dict]:
    """Ritorna la storia dei messaggi formattata per LiteLLM."""
    memory = get_session_history(session_id)
    formatted_history = []
    for m in memory.messages:
        if isinstance(m, SystemMessage):
            formatted_history.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            formatted_history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            formatted_history.append({"role": "assistant", "content": m.content})
    return formatted_history

def add_to_history(session_id: str, message: str, role: str = "user"):
    """Aggiunge un messaggio alla memoria della sessione."""
    memory = get_session_history(session_id)
    if role == "user":
        memory.add_message(HumanMessage(content=message))
    elif role == "assistant":
        memory.add_message(AIMessage(content=message))
