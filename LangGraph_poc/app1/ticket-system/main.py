"""CLI entry point for the Ticket Processing System"""

from graph import process_ticket


def print_result(state) -> None:
    print(f"\n{'='*60}")
    print(f"Ticket ID : {state.ticket_id}")
    print(f"Category  : {state.category} (confidence: {state.confidence:.0%})")
    print(f"Priority  : {state.priority} | SLA: {state.sla_hours}h")
    print(f"Department: {state.department}")
    print(f"Quality   : {state.quality_score:.0%} (retries: {state.retry_count})")
    print(f"\nResponse:\n  {state.response}")
    print(f"\nAudit trail:")
    for h in state.history:
        print(f"  [{h['time'][11:19]}] {h['node']:20s} {h['detail']}")
    print("="*60)


if __name__ == "__main__":
    tickets = [
        ("TKT-001", "My payment was charged twice this month and I need a refund immediately!"),
        ("TKT-002", "The app crashes when I try to upload a file larger than 10MB."),
        ("TKT-003", "Can you add dark mode to the dashboard?"),
    ]

    for ticket_id, message in tickets:
        print(f"\nProcessing {ticket_id}: {message[:60]}...")
        result = process_ticket(ticket_id, message)
        print_result(result)
