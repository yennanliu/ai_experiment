from typing import TypedDict


class AgentState(TypedDict):
    inbound_email: str
    email_type: str
    template: str
    required_fields: list[str]
    draft_reply: str
    checklist: list[str]
