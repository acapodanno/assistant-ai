from pydantic import BaseModel, Field
from typing import List, Literal

class AgentResponse(BaseModel):
    answer: str = Field(
        description="La risposta testuale finale da mostrare al cliente, formulata in modo chiaro ed empatico."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Il livello di confidenza dell'agente nel fornire la risposta."
    )
    sources: List[str] = Field(
        default=[],
        description="Elenco delle fonti (es. file della knowledge base) utilizzate per rispondere."
    )
    needs_human: bool = Field(
        default=False,
        description="Impostare a true se la richiesta richiede l'intervento di un operatore umano (es. escalation o reclamo)."
    )
