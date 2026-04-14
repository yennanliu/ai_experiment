"""
Unit tests for agent/nodes.py.

Each node is a pure function over AgentState dicts, so we mock
get_client() (OpenAI) and retrieve() (ChromaDB) to test them in isolation.
"""
from unittest.mock import MagicMock, patch
import pytest

from agent.nodes import (
    classify,
    select_template,
    retrieve_examples,
    generate_draft,
    build_checklist,
    score_draft,
)
from agent.state import AgentState
from agent.templates import TEMPLATES


# ── Helpers ───────────────────────────────────────────────────────────────────

def base_state(**overrides) -> AgentState:
    state: AgentState = {
        "inbound_email": "Test email content.",
        "email_type": "",
        "template": "",
        "required_fields": [],
        "draft_reply": "",
        "checklist": [],
        "retrieved_examples": [],
        "confidence_score": 0,
        "confidence_reason": "",
    }
    state.update(overrides)
    return state


def mock_completion(content: str) -> MagicMock:
    """Return a fake openai ChatCompletion object with a single choice."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── classify ──────────────────────────────────────────────────────────────────

class TestClassify:
    def test_known_type_is_set(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion("New PO")
            result = classify(base_state())
        assert result["email_type"] == "New PO"

    def test_unknown_llm_response_falls_back_to_other(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion("Gibberish Category")
            result = classify(base_state())
        assert result["email_type"] == "Other"

    @pytest.mark.parametrize("email_type", [
        "New PO", "Change Order", "Wrong Order / Quality Issue",
        "Inventory Inquiry", "Shipment Inquiry", "New Quote", "Other",
    ])
    def test_all_valid_types_pass_through(self, email_type):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(email_type)
            result = classify(base_state())
        assert result["email_type"] == email_type

    def test_other_state_fields_are_preserved(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion("Change Order")
            result = classify(base_state(draft_reply="existing draft", confidence_score=42))
        assert result["draft_reply"] == "existing draft"
        assert result["confidence_score"] == 42


# ── select_template ───────────────────────────────────────────────────────────

class TestSelectTemplate:
    def test_known_type_loads_correct_template(self):
        result = select_template(base_state(email_type="New PO"))
        assert result["template"] == TEMPLATES["New PO"]["template"]
        assert result["required_fields"] == TEMPLATES["New PO"]["required_fields"]

    def test_unknown_type_falls_back_to_other(self):
        result = select_template(base_state(email_type="Unknown Type XYZ"))
        assert result["template"] == TEMPLATES["Other"]["template"]

    @pytest.mark.parametrize("email_type", list(TEMPLATES.keys()))
    def test_all_email_types_return_non_empty_template(self, email_type):
        result = select_template(base_state(email_type=email_type))
        assert result["template"] != ""


# ── retrieve_examples ─────────────────────────────────────────────────────────

class TestRetrieveExamples:
    def test_examples_from_retriever_are_stored_in_state(self):
        fake_examples = [
            {"email": "e1", "reply": "r1", "email_type": "New PO"},
            {"email": "e2", "reply": "r2", "email_type": "New PO"},
        ]
        with patch("agent.nodes.retrieve", return_value=fake_examples):
            result = retrieve_examples(base_state(email_type="New PO"))
        assert result["retrieved_examples"] == fake_examples

    def test_empty_retriever_result_stored_as_empty_list(self):
        with patch("agent.nodes.retrieve", return_value=[]):
            result = retrieve_examples(base_state(email_type="Other"))
        assert result["retrieved_examples"] == []

    def test_retriever_called_with_correct_args(self):
        with patch("agent.nodes.retrieve", return_value=[]) as mock_retrieve:
            retrieve_examples(base_state(inbound_email="My email", email_type="New Quote"))
        mock_retrieve.assert_called_once_with("My email", "New Quote")


# ── generate_draft ────────────────────────────────────────────────────────────

class TestGenerateDraft:
    def test_draft_reply_is_set_from_llm(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion("Draft text.")
            result = generate_draft(base_state(template=TEMPLATES["New PO"]["template"]))
        assert result["draft_reply"] == "Draft text."

    def test_few_shot_examples_injected_into_system_prompt(self):
        examples = [{"email": "ref email", "reply": "ref reply", "email_type": "New PO"}]
        captured = {}

        def fake_create(**kwargs):
            captured["messages"] = kwargs["messages"]
            return mock_completion("Draft.")

        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.side_effect = fake_create
            generate_draft(base_state(
                template=TEMPLATES["New PO"]["template"],
                retrieved_examples=examples,
            ))

        system_content = captured["messages"][0]["content"]
        assert "ref reply" in system_content

    def test_no_examples_produces_no_few_shot_block(self):
        captured = {}

        def fake_create(**kwargs):
            captured["messages"] = kwargs["messages"]
            return mock_completion("Draft.")

        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.side_effect = fake_create
            generate_draft(base_state(
                template=TEMPLATES["Other"]["template"],
                retrieved_examples=[],
            ))

        system_content = captured["messages"][0]["content"]
        assert "historical replies" not in system_content


# ── build_checklist ───────────────────────────────────────────────────────────

class TestBuildChecklist:
    def test_missing_fields_added(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "Missing: PO number\nMissing: quantity"
            )
            result = build_checklist(base_state(
                inbound_email="Please send items.",
                required_fields=["PO number", "quantity"],
            ))
        missing = [i for i in result["checklist"] if i.startswith("Missing:")]
        assert len(missing) == 2
        assert any("PO number" in i for i in missing)
        assert any("quantity" in i for i in missing)

    def test_all_fields_present_yields_no_missing_items(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "All required fields present"
            )
            result = build_checklist(base_state(inbound_email="PO 1234, qty 100, due 06/01"))
        missing = [i for i in result["checklist"] if i.startswith("Missing:")]
        assert missing == []

    def test_risk_keywords_are_flagged(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "All required fields present"
            )
            result = build_checklist(base_state(
                inbound_email="This is HOT and urgent, ASAP please."
            ))
        risk_flags = [i for i in result["checklist"] if i.startswith("Risk flag:")]
        assert len(risk_flags) >= 1

    def test_clean_email_has_no_risk_flags(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "All required fields present"
            )
            result = build_checklist(base_state(
                inbound_email="PO 5660, part A1234, quantity 100, due 05/15."
            ))
        risk_flags = [i for i in result["checklist"] if i.startswith("Risk flag:")]
        assert risk_flags == []

    def test_risk_check_is_case_insensitive(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "All required fields present"
            )
            result = build_checklist(base_state(inbound_email="this is a qc hold situation."))
        assert any("QC hold" in i for i in result["checklist"])


# ── score_draft ───────────────────────────────────────────────────────────────

class TestScoreDraft:
    def test_score_and_reason_are_parsed(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "SCORE: 88\nREASON: Draft correctly extracts PO details and uses proper tone."
            )
            result = score_draft(base_state(draft_reply="Thank you for PO 1234."))
        assert result["confidence_score"] == 88
        assert "PO details" in result["confidence_reason"]

    def test_malformed_score_falls_back_to_70(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "SCORE: not-a-number\nREASON: Something."
            )
            result = score_draft(base_state(draft_reply="Draft."))
        assert result["confidence_score"] == 70

    def test_score_boundary_100(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "SCORE: 100\nREASON: Perfect reply."
            )
            result = score_draft(base_state(draft_reply="Draft."))
        assert result["confidence_score"] == 100

    def test_score_boundary_0(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "SCORE: 0\nREASON: Completely wrong."
            )
            result = score_draft(base_state(draft_reply="Draft."))
        assert result["confidence_score"] == 0

    def test_missing_reason_line_yields_empty_string(self):
        with patch("agent.nodes.get_client") as mock_get:
            mock_get.return_value.chat.completions.create.return_value = mock_completion(
                "SCORE: 75"
            )
            result = score_draft(base_state(draft_reply="Draft."))
        assert result["confidence_reason"] == ""
        assert result["confidence_score"] == 75
