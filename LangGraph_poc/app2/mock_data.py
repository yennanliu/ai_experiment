# Mock emails covering common B2B manufacturer email scenarios
# All cases derived from email_example.pdf

MOCK_EMAILS = [
    # ── New PO ────────────────────────────────────────────────────────────────
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
        "label": "New PO — Single PO, fill 40HQ container",
        "email": (
            "Subject: PO 5780\n\n"
            "Hi,\n\n"
            "Attached is our PO 5780.\n\n"
            "OK to change the quantity of Food Strainers on this order to fill a 40FT HQ container.\n\n"
            "Please confirm ship date and ETA.\n\n"
            "Thanks,\nEllen"
        ),
    },
    {
        "label": "New PO — Urgent single item, direct ship to customer",
        "email": (
            "Subject: PO 5820\n\n"
            "Hi,\n\n"
            "Please find our PO 5820 attached. Item: B666-001, qty 500, due 06/01.\n\n"
            "Note: This shipment is shipping directly to our end customer.\n"
            "They want to use their own shipping company.\n"
            "I will send you the contact information for their agent in Taiwan shortly.\n\n"
            "Thanks,\nSarah"
        ),
    },

    # ── Change Order ──────────────────────────────────────────────────────────
    {
        "label": "Change Order — Revised PO, updated due date",
        "email": (
            "Subject: Revised PO 5660\n\n"
            "Dear team,\n\n"
            "Please see the revised PO 5660 attached.\n"
            "We have updated the due date from 05/15 to 06/01 due to our production schedule change.\n\n"
            "Please arrange the shipment accordingly.\n\n"
            "Best,\nAnthony"
        ),
    },
    {
        "label": "Change Order — OEM PO revision, same item",
        "email": (
            "Subject: RE: OEM PO #00050080\n\n"
            "Dear team,\n\n"
            "Thank you for the update. Please see revised PO#00050080 attached.\n"
            "Quantity updated from 200 to 350 pcs. Due date remains 05/30.\n\n"
            "Please arrange the shipment accordingly.\n\n"
            "Best,\nEllen"
        ),
    },

    # ── Wrong Order / Quality Issue ───────────────────────────────────────────
    {
        "label": "Wrong Order — Mislabeled box, QC hold",
        "email": (
            "Subject: 35 wrong pick up tubes\n\n"
            "FYI...we just realized in the B700XXXX box was instead the B667XXXX.\n\n"
            "We have at least 24 wrong ones.\n\n"
            "Can you investigate what happened?\n\n"
            "We also found 1 box, X24 that the box was labeled B700XXX, "
            "but had the B667XXX's in the box and that has been pulled and is also on QC hold.\n\n"
            "Please advise."
        ),
    },
    {
        "label": "Wrong Order — Wrong item delivered, need urgent replacement",
        "email": (
            "Subject: Wrong shipment — PO 5701\n\n"
            "Hi,\n\n"
            "Our shipment for PO 5701 arrived yesterday but the wrong items were delivered.\n\n"
            "We ordered: B450031, qty 100\n"
            "We received: B450032, qty 100\n\n"
            "This is causing a production delay. "
            "Could you please confirm which container this came from and advise next steps?\n\n"
            "Regards,\nMike"
        ),
    },

    # ── Inventory Inquiry ─────────────────────────────────────────────────────
    {
        "label": "Inventory Inquiry — Low stock, need full delivery schedule",
        "email": (
            "Subject: B666 XXX inventory\n\n"
            "Hi,\n\n"
            "B666 XXX — please confirm when I will receive more inventory...super low on hand.\n\n"
            "Thanks"
        ),
    },
    {
        "label": "Inventory Inquiry — Multiple parts, checking delivery status",
        "email": (
            "Subject: Inventory check — B450031 & A450031\n\n"
            "Hi,\n\n"
            "Can you please give us an update on when we can expect delivery on:\n"
            "- B450031 (PO 5661, 300 pcs due 05/20)\n"
            "- A450031 (PO 5661, 300 pcs due 06/20)\n\n"
            "We are running low and need to plan our production schedule.\n\n"
            "Thanks,\nSarah"
        ),
    },

    # ── Shipment Inquiry ──────────────────────────────────────────────────────
    {
        "label": "Shipment Inquiry — Ready date + customer's own freight agent",
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
        "label": "Shipment Inquiry — Blodgett POs, contact change",
        "email": (
            "Subject: Blodgett purchase orders 3/17\n\n"
            "Hi,\n\n"
            "Just a heads up — going forward, please copy both myself and Casey "
            "on all communications related to the Blodgett purchase orders.\n\n"
            "Casey will be taking over as the main point of contact starting next month.\n\n"
            "Thanks,\nAnthony"
        ),
    },

    # ── New Quote ─────────────────────────────────────────────────────────────
    {
        "label": "New Quote — New part, request price + FA/production date",
        "email": (
            "Subject: Quote ECO 3100 / New PO 6001\n\n"
            "Hi,...need quote and FA/production date for 100 units please.\n\n"
            "New part: 120-00045 PUMP RELIEF TUBE.\n"
            "Drawing attached.\n\n"
            "Thanks"
        ),
    },
    {
        "label": "New Quote — Tooling + sample request for new component",
        "email": (
            "Subject: New part quote — ECO 3155\n\n"
            "Hi,\n\n"
            "Please provide a quote for the new part ECO 3155, VALVE SEAT RING.\n"
            "Quantity needed: 200 pcs for production, plus 3 samples for approval.\n\n"
            "Please include tooling cost and expected FA date.\n\n"
            "Thanks,\nDustin"
        ),
    },
]
