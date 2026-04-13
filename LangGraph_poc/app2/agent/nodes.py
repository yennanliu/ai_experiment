from __future__ import annotations
from functools import lru_cache
from openai import OpenAI
from .templates import TEMPLATES, RISK_KEYWORDS
from .state import AgentState


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI()

EMAIL_TYPES = ["New PO", "Change Order", "Wrong Order / Quality Issue", "Inventory Inquiry", "Shipment Inquiry", "New Quote", "Other"]


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
                    "You are an email assistant for a Taiwan-based B2B manufacturer. "
                    "Your job is to generate a reply email using the provided template. "
                    "Rules:\n"
                    "- Tone: professional, concise, factual — no unnecessary filler\n"
                    "- Extract all values (PO numbers, part numbers, quantities, dates) directly from the inbound email\n"
                    "- Where info is not provided, keep the [XXX] placeholder as-is — the user will fill these in\n"
                    "- For HOT/urgent items, mark them clearly with (HOT)\n"
                    "- For partial shipments, use bullet points as shown in the template\n"
                    "- Always offer proforma invoice for New PO and Change Order types\n"
                    "- Do NOT invent shipping dates, freight costs, or quantities — use [XXX] if unknown\n\n"
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
                    "You are reviewing an inbound business email for a Taiwan manufacturer. "
                    f"Check if the following required fields are present or inferable: {', '.join(state['required_fields'])}. "
                    "For each field that is missing or unclear and cannot be inferred, output one line: 'Missing: <field name>'. "
                    "If nothing is missing, output 'All required fields present'. "
                    "Note: shipping dates, freight costs, and production details are NOT expected in the inbound email — do not flag those. "
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
