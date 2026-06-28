from src.rag.run_rag import run_rag
import json
from src.agent.models.ticket import CustomerInfo, IssueType, OrderDetails, Ticket
from loguru import logger


def get_order_by_id(order_id: str) -> dict | None:
    logger.info(f"Fetching order by ID: {order_id}")
    try:
        with open("./data/orders.json", "r") as f:
            orders = json.load(f)
        for order in orders:
            if order["order_id"] == order_id:
                logger.info(f"Order {order_id} found.")
                return order
        logger.info(f"Order {order_id} not found.")
        return None
    except FileNotFoundError:
        logger.error("Order database not found.")
        return {"error": "Database ordini non trovato."}
    except json.JSONDecodeError:
        logger.error("Order database is corrupted.")
        return {"error": "Database ordini corrotto."}


def create_ticket(
    customer: dict,
    order_id: str,
    issue_type: str,
    issue_description: str,
    order_details: dict | None = None
) -> dict:
    logger.info(f"Creating ticket for order {order_id} (Issue: {issue_type})")
    try:
        with open("./data/tickets.json", "r") as f:
            tickets = json.load(f)
    except FileNotFoundError:
        logger.warning("Tickets database not found, creating a new one.")
        tickets = []  # Crea la lista se il file non esiste
    except json.JSONDecodeError:
        logger.error("Tickets database is corrupted.")
        return {"error": "Database ticket corrotto."}

    # Costruiamo il ticket validato con Pydantic
    try:
        ticket = Ticket(
            ticket_id=len(tickets) + 1,
            customer=CustomerInfo(**customer),
            order_id=order_id,
            issue_type=IssueType(issue_type),
            issue_description=issue_description,
            order_details=OrderDetails(**(order_details or {}))
        )
    except Exception as e:
        logger.error(f"Invalid ticket data: {str(e)}")
        return {"error": f"Dati ticket non validi: {str(e)}"}

    ticket_dict = ticket.model_dump(mode="json")
    tickets.append(ticket_dict)

    try:
        with open("./data/tickets.json", "w") as f:
            json.dump(tickets, f, indent=4, ensure_ascii=False)
        logger.info(f"Ticket #{ticket.ticket_id} saved successfully.")
    except Exception as e:
        logger.error(f"Unable to save ticket: {str(e)}")
        return {"error": f"Impossibile salvare il ticket: {str(e)}"}

    return {
        "ticket_id": ticket.ticket_id,
        "status": ticket.status.value,
        "message": f"Ticket #{ticket.ticket_id} aperto con successo per {ticket.customer.name} — tipo: {ticket.issue_type.value}."
    }


def rag_knowledge_base(query: str) -> dict:
    """
    Cerca nella Knowledge Base di GreenThumb e ritorna un dict JSON-serializzabile
    con la risposta e i nomi dei file sorgente usati.
    """
    logger.info(f"Querying RAG Knowledge Base with: '{query}'")
    result = run_rag(query)
    # Estrai i nomi file dai documenti (non serializzabili come oggetti)
    sources = [
        doc.metadata.get("source", "").split("/")[-1]
        for doc in result.get("context", [])
    ]
    unique_sources = list(dict.fromkeys(sources))
    logger.info(f"RAG query completed. Found sources: {unique_sources}")
    return {
        "answer": result.get("answer", ""),
        "sources": unique_sources  # deduplica mantenendo l'ordine
    }