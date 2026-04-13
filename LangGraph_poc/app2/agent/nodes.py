from __future__ import annotations
from functools import lru_cache
from openai import OpenAI
from .templates import TEMPLATES, RISK_KEYWORDS
from .state import AgentState


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI()

EMAIL_TYPES = ["New PO", "Change Order", "Wrong Order", "Other"]


def classify(state: AgentState) -> AgentState:
    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Classify the following email into exactly one of these categories: {', '.join(EMAIL_TYPES)}. "
                    "Reply with only the category name, nothing else."
                ),
            },
            {"role": "user", "content": state["inbound_email"]},
        ],
        temperature=0,
    )
    email_type = response.choices[0].message.content.strip()
    if email_type not in EMAIL_TYPES:
        email_type = "Other"
    return {**state, "email_type": email_type}


def select_template(state: AgentState) -> AgentState:
    template_data = TEMPLATES.get(state["email_type"], TEMPLATES["Other"])
    return {**state, "template": template_data["template"], "required_fields": template_data["required_fields"]}


def generate_draft(state: AgentState) -> AgentState:
    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional business email assistant. "
                    "Generate a reply email by filling in the provided template based on the inbound email context. "
                    "Keep the tone professional, concise, and consistent. "
                    "Replace all placeholders like [Customer Name], [PO Number], etc. with values extracted from the inbound email. "
                    "If a value cannot be determined, keep the placeholder as-is.\n\n"
                    f"Template to use:\n{state['template']}"
                ),
            },
            {
                "role": "user",
                "content": f"Inbound email:\n{state['inbound_email']}",
            },
        ],
        temperature=0.3,
    )
    draft = response.choices[0].message.content.strip()
    return {**state, "draft_reply": draft}


def build_checklist(state: AgentState) -> AgentState:
    checklist = []

    # Check for missing required fields
    response = get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are reviewing an inbound email to flag missing information. "
                    f"Check if the following required fields are present: {', '.join(state['required_fields'])}. "
                    "For each field that is missing or unclear, output one line: 'Missing: <field name>'. "
                    "If nothing is missing, output 'All required fields present'. "
                    "Reply with only the checklist lines, nothing else."
                ),
            },
            {"role": "user", "content": state["inbound_email"]},
        ],
        temperature=0,
    )
    missing = response.choices[0].message.content.strip()
    if missing != "All required fields present":
        checklist.extend([line for line in missing.splitlines() if line.startswith("Missing:")])

    # Rule-based risk keyword scan
    email_lower = state["inbound_email"].lower()
    for keyword in RISK_KEYWORDS:
        if keyword.lower() in email_lower:
            checklist.append(f"Risk flag: '{keyword}' detected in email")

    return {**state, "checklist": checklist}
