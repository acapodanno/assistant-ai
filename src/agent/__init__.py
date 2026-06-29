# Agent package
# Espone i componenti pubblici dell'agente e della memoria

from .react_agent import GreenThumbAgent
from .tools import get_order_by_id, create_ticket
from .models.agent_response import AgentResponse
from .models.ticket import (
    Ticket,
    CustomerInfo,
    OrderDetails,
    IssueType,
    TicketStatus,
)
from .memory import get_formatted_history, add_to_history

__all__ = [
    "GreenThumbAgent",
    "get_order_by_id",
    "create_ticket",
    "AgentResponse",
    "Ticket",
    "CustomerInfo",
    "OrderDetails",
    "IssueType",
    "TicketStatus",
    "get_formatted_history",
    "add_to_history",
]
