from src.rag.run_rag import run_rag
import json
from src.agent.models.ticket import CustomerInfo, IssueType, OrderDetails, Ticket
from loguru import logger

def get_order_by_id(order_id: str) -> dict | None:
    logger.info(f"Fetching order: {order_id}")
    try:
        with open("./data/orders.json", "r") as f:
            orders = json.load(f)
        for order in orders:
            if order["order_id"] == order_id:
                return order
        return None
    except Exception:
        return None

def create_ticket(customer: dict, order_id: str, issue_type: str, issue_description: str, order_details: dict | None = None) -> dict:
    logger.info(f"Creating ticket for {order_id}")
    try:
        with open("./data/tickets.json", "r") as f:
            tickets = json.load(f)
    except Exception:
        tickets = []

    try:
        ticket = Ticket(
            ticket_id=len(tickets) + 1,
            customer=CustomerInfo(**customer),
            order_id=order_id,
            issue_type=IssueType(issue_type),
            issue_description=issue_description,
            order_details=OrderDetails(**(order_details or {}))
        )
        ticket_dict = ticket.model_dump(mode="json")
        tickets.append(ticket_dict)
        with open("./data/tickets.json", "w") as f:
            json.dump(tickets, f, indent=4, ensure_ascii=False)
        return {"ticket_id": ticket.ticket_id, "status": ticket.status.value, "message": "Successo"}
    except Exception as e:
        return {"error": str(e)}

def rag_knowledge_base(query: str) -> dict:
    result = run_rag(query)
    sources = [doc.metadata.get("source", "").split("/")[-1] for doc in result.get("context", [])]
    return {"answer": result.get("answer", ""), "sources": list(dict.fromkeys(sources))}
