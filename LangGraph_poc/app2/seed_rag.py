"""
Seed ChromaDB with historical email→reply examples.
Based on real patterns from email_example.pdf.

Usage:
    uv run python seed_rag.py          # seed all examples
    uv run python seed_rag.py --reset  # wipe collection and re-seed
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import chromadb
from agent.retriever import get_collection, add_example, count

EXAMPLES = [
    # ── New PO ────────────────────────────────────────────────────────────────
    {
        "id": "new-po-001",
        "email_type": "New PO",
        "email": (
            "New PO 5660 5661 5662\n"
            "5660 and 5662 are hot...please"
        ),
        "reply": (
            "Thank you for the new POs 5660 5661 5662.\n\n"
            "PO 5660, 140-XXXX (HOT)\n"
            "80 DUE XX/XX/XX —Ship in partial\n"
            "• We will not finish and ship 60 pcs via SEA until XX/XX/XX, due about XX/XX/XX.\n"
            "• For the balance 20 pcs, we will not finish and ship via SEA until XX/XX/XX, "
            "due about XX/XX/XX. If ship via AIR, we can ship via UPS PLT on XX/XX/XX, "
            "due about XX/XX/XX. The estimated freight cost is about US$ XXXXX. "
            "Please advise the shipping method.\n\n"
            "PO 5661, A4500XXX\n"
            "300 DUE XX/XX/XX —OK\n"
            "300 DUE XX/XX/XX —OK\n\n"
            "PO 5662, B45XXXX (HOT)\n"
            "100 DUE XX/XX/XX —Ship in partial\n"
            "• We can ship 80 pcs via SEA on XX/XX, due about XX/XX/XX.\n"
            "• For the balance 20 pcs, we will finish and ship via SEA until XX/XX/XX, "
            "due about XX/XX/XX. If ship via AIR, we can ship via UPS on XX/XX/XX, "
            "due about XX/XX/XX. The estimated freight cost is about US$ XXX.XX. "
            "Please advise the shipping method.\n\n"
            "Attached please find the proforma invoice for your reference."
        ),
    },
    {
        "id": "new-po-002",
        "email_type": "New PO",
        "email": (
            "PO 5780\n"
            "Attached is our PO 5780.\n"
            "OK to change the quantity of Food Strainers on this order to fill a 40FT HQ container."
        ),
        "reply": (
            "Thank you for the new PO 5780.\n\n"
            "Please note that we adjust the quantity from 2500 to 1960 pcs in order to fill in "
            "a 40 HQ container. We will ship this order via SEA on XX/XX/XX, the ETA XXX CITY "
            "is on about XX/XX/XX. Attached please find the proforma invoice for your reference. "
            "If convenient, we hope to receive revised PO 5780 from you."
        ),
    },
    # ── Change Order ──────────────────────────────────────────────────────────
    {
        "id": "change-001",
        "email_type": "Change Order",
        "email": "Revised PO 5660 — updated due date from 05/15 to 06/01.",
        "reply": (
            "Thank you for the revised PO 5660.\n\n"
            "We will arrange the shipment accordingly."
        ),
    },
    {
        "id": "change-002",
        "email_type": "Change Order",
        "email": "RE: OEM PO #00050080 — please see revised PO attached. Quantity updated from 200 to 350 pcs.",
        "reply": (
            "Thank you for the revised PO#00050080.\n\n"
            "We will arrange the shipment accordingly."
        ),
    },
    # ── Wrong Order / Quality Issue ───────────────────────────────────────────
    {
        "id": "wrong-001",
        "email_type": "Wrong Order / Quality Issue",
        "email": (
            "35 wrong pick up tubes.\n"
            "FYI...we just realized in the B700XXXX box was instead the B667XXXX.\n"
            "We have at least 24. Can you investigate what happened?\n"
            "We found 1 box, X24 that the box was labeled B700XXX, but had the B667XXX's "
            "in the box and that has been pulled and is also on QC hold."
        ),
        "reply": (
            "We are sorry to hear about this issue.\n\n"
            "To better understand the issue, we would like to confirm whether this box is "
            "from container CE-1140XXXX (received in XXXXXX 2025), or if it came from your existing stock.\n\n"
            "Could you also please provide us with the photos of the outer carton, "
            "the dimensions and the weight of carton for our reference?\n\n"
            "Thank you for your assistance. We are sorry for the inconvenience caused."
        ),
    },
    # ── Inventory Inquiry ─────────────────────────────────────────────────────
    {
        "id": "inventory-001",
        "email_type": "Inventory Inquiry",
        "email": (
            "B666 XXX\n"
            "Please confirm when I will receive more inventory...super low on hand."
        ),
        "reply": (
            "Thank you for informing us that you are low on B666 XXX. "
            "The current delivery schedule is as follows for your reference.\n\n"
            "PO 4XXX, B666 XXX\n"
            "300 DUE XX/XX/XX —Ship via UPS PLT. Air is confirmed. We can ship in partial shipments below.\n"
            "• We have stock of 20 pcs, and we can ship early on XX/XX via UPS, due about XX/XX.\n"
            "• For the balance XXX pcs, due to the late of materials, we will finish and ship via "
            "UPS PLT on XX/XX as planned, due about XX/XX.\n\n"
            "700 DUE XX/XX/XX —See below.\n"
            "• Originally, we planned to ship via SEA on XX/XX, due about XX/XX. Due to material delay, "
            "we will not finish until XX/XX, and ship via SEA, due about XX/XX.\n"
            "• If you need to ship a partial from the sea shipment to the air shipment, we can ship on "
            "XX/XX via UPS PLT, due about XX/XX. This item is 20 pcs/CTN. Please advise the quantity "
            "that you need by air, and we will provide the estimated air freight cost for your confirmation.\n\n"
            "PO 2XXX, B666XXXX\n"
            "1380 DUE XX/XX/XX —Shipped 1380, CE-114XXXX, due about 1/22.\n\n"
            "PO 5XXX, B666XXXX\n"
            "1000 DUE XX/XX/XX —We will ship via SEA on 2/3, due about 4/6."
        ),
    },
    # ── Shipment Inquiry ──────────────────────────────────────────────────────
    {
        "id": "shipment-001",
        "email_type": "Shipment Inquiry",
        "email": (
            "PO 5XXX\n"
            "Please advise the date that PO 5XXX will be ready to ship.\n"
            "This shipment is shipping directly to our customer and they want to use their shipping company.\n"
            "I will provide the contact information for their agent in Taiwan soon."
        ),
        "reply": (
            "Thank you for your email.\n\n"
            "Regarding PO 5XXX, this shipment is estimated to be ready on XX/XX.\n\n"
            "We will wait for the contact information of the agent from you."
        ),
    },
    {
        "id": "shipment-002",
        "email_type": "Shipment Inquiry",
        "email": "Blodgett purchase orders 3/17 — please copy both myself and Casey on all future communications.",
        "reply": (
            "Thank you for your information.\n\n"
            "We will contact you and Casey in the future."
        ),
    },
    # ── New Quote ─────────────────────────────────────────────────────────────
    {
        "id": "quote-001-day1",
        "email_type": "New Quote",
        "email": (
            "Quote ECO 31XXX - New PO 6XXX\n"
            "Hi,...need quote and FA/production date for 100 units please."
        ),
        "reply": (
            "Thank you for providing drawing and the quantity needed for the new part "
            "120-000XXX. PUMP RELIEF TUBE.\n\n"
            "We will evaluate and will provide the quote and FA/production date for 100 units on 2/13.\n\n"
            "Thank you for your patience."
        ),
    },
    {
        "id": "quote-001-day2",
        "email_type": "New Quote",
        "email": (
            "Quote ECO 31XXX - following up on quote and FA/production date for 100 units."
        ),
        "reply": (
            "Regarding the new part 120-000XXX.-001, the following is the price for your reference.\n\n"
            "CIF XXX.\n"
            "120-00XXX, US$ XXX/pc\n\n"
            "The tooling charge: US$ XXX/set, which can be spread over the first 300 pcs.\n"
            "(The unit price for the first XXX pcs will be US$XXX/pc.)\n\n"
            "If the price is acceptable, we can send 2 samples on X/X, due about X/X.\n\n"
            "If the sample is approved by X/X, we can ship the production for 100 on X/X, due about 7/6."
        ),
    },
]


def seed(reset: bool = False) -> None:
    if reset:
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            client.delete_collection("email_examples")
            print("Collection reset.")
        except Exception:
            pass
        # reset lru_cache so get_collection re-creates it
        get_collection.cache_clear()

    col = get_collection()
    existing_ids: set[str] = set(col.get()["ids"]) if col.count() > 0 else set()

    inserted = updated = 0
    for ex in EXAMPLES:
        is_new = ex["id"] not in existing_ids
        add_example(
            doc_id=ex["id"],
            inbound_email=ex["email"],
            reply=ex["reply"],
            email_type=ex["email_type"],
        )
        if is_new:
            inserted += 1
            print(f"  [NEW]     {ex['id']}")
        else:
            updated += 1
            print(f"  [UPSERT]  {ex['id']}")

    print(f"\nDone: {inserted} inserted, {updated} already existed (total {col.count()} in collection)")
    print("\nBreakdown:")
    from collections import Counter
    c = Counter(e["email_type"] for e in EXAMPLES)
    for t, n in sorted(c.items()):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    seed(reset="--reset" in sys.argv)
