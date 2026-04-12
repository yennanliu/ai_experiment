"""LangGraph state definition"""

from dataclasses import dataclass, field


@dataclass
class TicketState:
    # Input
    ticket_id: str = ""
    user_message: str = ""

    # Classification
    category: str = ""      # technical | billing | account | feature_request | bug_report | other
    confidence: float = 0.0

    # Priority
    priority: str = ""      # critical | high | medium | low
    sla_hours: int = 24

    # Routing
    department: str = ""

    # Response
    response: str = ""
    quality_score: float = 0.0
    retry_count: int = 0

    # Audit
    history: list = field(default_factory=list)
