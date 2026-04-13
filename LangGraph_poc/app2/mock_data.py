# Mock emails covering common B2B manufacturer email scenarios
# XXX = placeholders the user fills in manually before sending

MOCK_EMAILS = [
    {
        "label": "New PO — Multiple POs, two marked HOT",
        "email": (
            "Subject: New PO 5660 5661 5662\n\n"
            "Hi,\n\n"
            "Please find attached our new purchase orders PO 5660, PO 5661, and PO 5662.\n\n"
            "PO 5660 — 140-2201 (HOT), qty 80, due 05/15\n"
            "PO 5661 — A450031, qty 300, due 05/20; qty 300, due 06/20\n"
            "PO 5662 — B450078 (HOT), qty 100, due 05/10\n\n"
            "PO 5660 and PO 5662 are hot...please expedite.\n\n"
            "Thanks,\nDustin"
        ),
    },
    {
        "label": "Change Order — Qty adjusted to fill 40HQ container",
        "email": (
            "Subject: PO 5780 — quantity change\n\n"
            "Hi,\n\n"
            "Attached is our PO 5780.\n\n"
            "OK to change the quantity of Food Strainers on this order to fill a 40FT HQ container.\n\n"
            "Please confirm ship date and ETA.\n\n"
            "Thanks,\nEllen"
        ),
    },
    {
        "label": "Change Order — Revised PO date update",
        "email": (
            "Subject: Revised PO 5660\n\n"
            "Dear team,\n\n"
            "Please see the revised PO 5660 attached.\n"
            "We have updated the due date from 05/15 to 06/01.\n\n"
            "Please arrange the shipment accordingly.\n\n"
            "Best,\nAnthony"
        ),
    },
    {
        "label": "Wrong Order / Quality Issue — Mislabeled box",
        "email": (
            "Subject: 35 wrong pick up tubes\n\n"
            "FYI...we just realized in the B700XXXX box was instead the B667XXXX.\n\n"
            "We have at least 24.\n\n"
            "Can you investigate what happened?\n\n"
            "And we found 1 box, X24 that the box was labeled B700XXX, but had the "
            "B667XXX's in the box and that has been pulled and is also on QC hold.\n\n"
            "Please advise.\n\n"
            "Thanks"
        ),
    },
    {
        "label": "Inventory Inquiry — Low stock, need delivery schedule",
        "email": (
            "Subject: B666 XXX inventory\n\n"
            "Hi,\n\n"
            "B666 XXX — please confirm when I will receive more inventory...super low on hand.\n\n"
            "Thanks"
        ),
    },
    {
        "label": "Shipment Inquiry — PO ready date + customer's own freight agent",
        "email": (
            "Subject: PO 5820 ship date\n\n"
            "Hi,\n\n"
            "Please advise the date that PO 5820 will be ready to ship.\n\n"
            "This shipment is shipping directly to our customer and they want to use their shipping company.\n\n"
            "I will provide the contact information for their agent in Taiwan soon.\n\n"
            "Thanks,\nSarah"
        ),
    },
    {
        "label": "New Quote — New part, request for price + FA date",
        "email": (
            "Subject: Quote ECO 3100 / New PO 6001\n\n"
            "Hi,...need quote and FA/production date for 100 units please.\n\n"
            "New part: 120-00045 PUMP RELIEF TUBE.\n"
            "Drawing attached.\n\n"
            "Thanks"
        ),
    },
]
