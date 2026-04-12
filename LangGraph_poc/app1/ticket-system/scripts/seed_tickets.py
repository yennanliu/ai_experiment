"""
Seed script — submits N random tickets to the running server.

Usage:
  uv run scripts/seed_tickets.py           # 100 tickets, default concurrency
  uv run scripts/seed_tickets.py --count 20 --concurrency 5 --url http://localhost:8000
"""

import argparse
import asyncio
import random
import sys
import time

import httpx

# ── Sample data ───────────────────────────────────────────────────────────────

TEMPLATES = [
    # billing
    "I was charged twice for my subscription this month. Please issue a refund.",
    "My invoice shows an incorrect amount. I need this corrected immediately.",
    "I cancelled my plan but was still billed. Can you fix this?",
    "I need a receipt for my last payment for tax purposes.",
    "Why did my plan price increase without any notice?",
    "I accidentally purchased the wrong plan. Can I get a refund?",
    "My free trial ended but I was charged before I could cancel.",
    # technical
    "The app crashes every time I try to upload a file larger than 10 MB.",
    "I can't log in — I get a '500 Internal Server Error' every time.",
    "The dashboard takes over 30 seconds to load. It used to be instant.",
    "Dark mode is broken — text is invisible on some screens.",
    "Notifications are not being delivered even though they are enabled.",
    "The mobile app freezes when I switch between tabs quickly.",
    "Copy/paste doesn't work in the text editor on Firefox.",
    "My data export is always empty even though I have records.",
    "Two-factor authentication keeps failing with a valid code.",
    "The search bar returns no results for keywords that clearly exist.",
    "Videos won't play — they just show a blank black screen.",
    # account
    "I forgot my password and the reset email never arrived.",
    "I need to change the email address on my account.",
    "My account was locked after too many login attempts. Please unlock it.",
    "How do I delete my account and all associated data?",
    "I want to transfer my account ownership to a colleague.",
    "Can I merge two accounts that I accidentally created?",
    "My profile picture won't update — it keeps reverting to the old one.",
    "I need to update my billing address on the account.",
    # feature_request
    "Please add support for keyboard shortcuts in the editor.",
    "It would be great to have a dark mode option in the settings.",
    "Can you add CSV export for the analytics dashboard?",
    "I'd like to be able to schedule reports to be emailed weekly.",
    "Please add two-factor authentication via hardware keys.",
    "It would help to have a bulk-delete option for old records.",
    "Can you support SSO login with our company's Okta account?",
    "Please add an API endpoint to retrieve historical usage stats.",
    "I'd love a mobile app for iOS.",
    "Can you integrate with Slack for real-time notifications?",
    # bug_report
    "Clicking 'Save' in settings does nothing — changes are lost on refresh.",
    "The date picker shows the wrong month when the timezone is set to UTC+9.",
    "Sorting the table by 'Date' puts items in the wrong order.",
    "The pagination breaks when filtering — it always resets to page 1.",
    "Deleting a record shows a success message but the record is still there.",
    "The chart legend overlaps the data when there are more than 5 series.",
    "Auto-save fires every second instead of every 30 seconds.",
    "The 'Forgot password' link on the login page is broken (404).",
    "Dragging items in the kanban board sometimes duplicates them.",
    "The tooltip on the pricing page shows the wrong plan name.",
    # other
    "Hi, I'm having trouble understanding how to use the API. Any docs?",
    "Do you offer a non-profit discount?",
    "I'd like to speak to a human support agent.",
    "Can I get an extension on my trial period?",
    "What are your data retention policies?",
    "Do you have a status page I can monitor?",
    "I need a W-9 form for vendor setup at my company.",
]

URGENCY_PREFIXES = [
    "", "", "",                         # most tickets are plain
    "URGENT: ",
    "Critical issue — ",
    "[Production down] ",
]


def random_message() -> str:
    prefix = random.choice(URGENCY_PREFIXES)
    body   = random.choice(TEMPLATES)
    return prefix + body


# ── Core logic ────────────────────────────────────────────────────────────────

async def send_ticket(client: httpx.AsyncClient, base_url: str, idx: int, total: int) -> dict:
    message = random_message()
    resp    = await client.post(f"{base_url}/tickets", json={"message": message})
    resp.raise_for_status()
    ticket = resp.json()
    print(f"  [{idx:>3}/{total}] {ticket['ticket_id']}  {message[:70]}")
    return ticket


async def seed(base_url: str, count: int, concurrency: int) -> None:
    print(f"\nSeeding {count} tickets → {base_url}  (concurrency={concurrency})\n")
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=10) as client:
        sem = asyncio.Semaphore(concurrency)

        async def guarded(idx):
            async with sem:
                try:
                    return await send_ticket(client, base_url, idx, count)
                except Exception as e:
                    print(f"  [{idx:>3}/{count}] ERROR: {e}", file=sys.stderr)
                    return None

        results = await asyncio.gather(*[guarded(i + 1) for i in range(count)])

    ok      = sum(1 for r in results if r)
    elapsed = time.perf_counter() - t0
    print(f"\nDone — {ok}/{count} submitted in {elapsed:.1f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed random tickets into the ticket system")
    parser.add_argument("--count",       type=int, default=100,                   help="Number of tickets to submit (default: 100)")
    parser.add_argument("--concurrency", type=int, default=10,                    help="Max parallel requests (default: 10)")
    parser.add_argument("--url",         type=str, default="http://localhost:8000", help="Server base URL")
    args = parser.parse_args()

    asyncio.run(seed(args.url, args.count, args.concurrency))


if __name__ == "__main__":
    main()
