"""PDF export tool — converts a ticket record to a PDF byte stream"""

from fpdf import FPDF


def ticket_to_pdf(ticket: dict) -> bytes:
    """Render a ticket record as a PDF and return raw bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Ticket Report: {ticket.get('ticket_id', '')}", ln=True)
    pdf.ln(2)

    # Submitted / status
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Submitted: {ticket.get('submitted_at', '')}   Status: {ticket.get('status', '')}", ln=True)
    pdf.ln(4)

    # Message
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "User Message", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, ticket.get("message", ""))
    pdf.ln(4)

    result = ticket.get("result")
    if result:
        # Metadata table
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Analysis", ln=True)
        pdf.set_font("Helvetica", "", 10)
        meta_fields = [
            ("Category",    result.get("category", "")),
            ("Confidence",  f"{int(result.get('confidence', 0) * 100)}%"),
            ("Priority",    result.get("priority", "")),
            ("SLA",         f"{result.get('sla_hours', '')}h"),
            ("Department",  result.get("department", "")),
            ("Quality",     f"{int(result.get('quality_score', 0) * 100)}%"),
            ("Retries",     str(result.get("retry_count", 0))),
        ]
        for label, value in meta_fields:
            pdf.cell(40, 6, f"{label}:", border=0)
            pdf.cell(0, 6, value, ln=True)
        pdf.ln(4)

        # Research notes
        if result.get("research_notes"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Research Notes", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, result["research_notes"])
            pdf.ln(4)

        # Response
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Agent Response", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, result.get("response", ""))
        pdf.ln(4)

        # Process log
        history = result.get("history", [])
        if history:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Process Log", ln=True)
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(80, 80, 80)
            for h in history:
                line = f"[{h.get('time','')[:19]}] {h.get('node',''):20s} {h.get('detail','')}"
                pdf.multi_cell(0, 5, line)

    return bytes(pdf.output())
