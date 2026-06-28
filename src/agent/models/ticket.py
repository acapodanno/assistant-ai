from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class IssueType(str, Enum):
    MANCATA_CONSEGNA = "MANCATA_CONSEGNA"
    PRODOTTO_DANNEGGIATO = "PRODOTTO_DANNEGGIATO"
    RESO = "RESO"
    RIMBORSO = "RIMBORSO"
    ALTRO = "ALTRO"


class TicketStatus(str, Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    CLOSED = "CLOSED"


class CustomerInfo(BaseModel):
    name: str
    email: str
    phone: str


class OrderDetails(BaseModel):
    products: list[dict] = Field(default_factory=list)
    total: Optional[float] = None
    shipping_address: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None


class Ticket(BaseModel):
    ticket_id: int
    customer: CustomerInfo
    order_id: str
    issue_type: IssueType
    issue_description: str
    order_details: OrderDetails = Field(default_factory=OrderDetails)
    status: TicketStatus = TicketStatus.OPEN
    created_at: datetime = Field(default_factory=datetime.now)
