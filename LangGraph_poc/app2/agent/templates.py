# Email types for a B2B manufacturer handling international trade orders
# Reply style: concise, professional, factual — no fluff
# XXX placeholders = values the user must fill in manually before sending

TEMPLATES: dict[str, dict] = {

    "New PO": {
        "required_fields": ["PO number(s)", "part number(s)", "quantity", "due date"],
        "template": """\
Thank you for the new PO(s) [PO numbers].

PO [PO number], [Part Number] [(HOT) if urgent]
[Qty] DUE [XX/XX/XX] —[OK / Ship in partial]
• We will [finish and / not finish and] ship [qty] pcs via [SEA/AIR] on [XX/XX], due about [XX/XX/XX].
• [For the balance [qty] pcs, we will [finish and / not finish and] ship via [SEA/AIR] until [XX/XX/XX], due about [XX/XX/XX]. If ship via AIR, we can ship via UPS [PLT] on [XX/XX/XX], due about [XX/XX/XX]. The estimated freight cost is about US$ [XXX]. Please advise the shipping method.]

[Repeat block for each PO line item]

Attached please find the proforma invoice for your reference.""",
    },

    "Change Order": {
        "required_fields": ["PO number", "change details (quantity / date / item)", "container type if relevant"],
        "template": """\
Thank you for the [revised PO / new PO] [PO number].

Please note that we [adjust the quantity from [XXX] to [XXX] pcs / update the due date to [XX/XX/XX]] [in order to fill in a [40HQ / 20FT] container].

We will ship this order via [SEA/AIR] on [XX/XX/XX], the ETA [CITY] is on about [XX/XX/XX].

Attached please find the proforma invoice for your reference. If convenient, we hope to receive [a revised PO / your confirmation] from you.""",
    },

    "Wrong Order / Quality Issue": {
        "required_fields": ["wrong item description", "correct item expected", "container or shipment reference if available"],
        "template": """\
We are sorry to hear about this issue.

To better understand the situation, we would like to confirm whether this [box / shipment] is from container [CE-XXXXXXX] (received in [month/year]), or if it came from your existing stock.

Could you also please provide us with:
• Photos of the outer carton
• The dimensions and weight of the carton

Thank you for your assistance. We are sorry for the inconvenience caused.""",
    },

    "Inventory Inquiry": {
        "required_fields": ["part number(s)", "PO number(s) if mentioned"],
        "template": """\
Thank you for informing us that you are low on [Part Number]. The current delivery schedule is as follows for your reference.

PO [XXXX], [Part Number]
[Qty] DUE [XX/XX/XX] —[Ship via UPS PLT / Ship via SEA]. [Air is confirmed / See below.]
• [We have stock of [XX] pcs, and we can ship early on [XX/XX] via [UPS/SEA], due about [XX/XX].]
• [For the balance [XXX] pcs, due to [late materials / production schedule], we will finish and ship via [UPS PLT / SEA] on [XX/XX] as planned, due about [XX/XX]. Please advise if you need any quantity by air.]

[Repeat for additional PO lines]

Please advise the shipping method and quantity needed by air if applicable.""",
    },

    "Shipment Inquiry": {
        "required_fields": ["PO number", "ready date if known", "special shipping arrangement if any"],
        "template": """\
Thank you for your email.

Regarding PO [XXXXX], this shipment is estimated to be ready on [XX/XX].

[We will wait for the contact information of the agent from you. / We will arrange shipment via [SEA/AIR] as discussed.] Please feel free to reach out if you have any further questions.""",
    },

    "New Quote": {
        "required_fields": ["part number / ECO number", "quantity", "drawing or spec reference"],
        "template": """\
Thank you for providing [the drawing and] the quantity needed for the new part [Part Number] — [Part Description].

We will evaluate and will provide the quote and FA/production date for [XXX] units on [XX/XX].

Thank you for your patience.

---

[Day 2 — after evaluation:]

Regarding the new part [Part Number], the following is the price for your reference.

CIF [CITY/PORT]:
[Part Number], US$ [XXX]/pc

The tooling charge: US$ [XXX]/set, which can be spread over the first [XXX] pcs.
(The unit price for the first [XXX] pcs will be US$ [XXX]/pc.)

If the price is acceptable, we can send [X] sample(s) on [XX/XX], due about [XX/XX].
If the sample is approved by [XX/XX], we can ship the production for [XXX] pcs on [XX/XX], due about [XX/XX].""",
    },

    "Other": {
        "required_fields": [],
        "template": """\
Thank you for your email.

We have received your message and will follow up accordingly. Please do not hesitate to contact us if you need any further assistance.""",
    },
}

RISK_KEYWORDS = [
    "QC hold", "quality hold", "defect", "wrong", "mislabeled",
    "hot", "urgent", "ASAP", "rush",
    "air freight", "UPS PLT", "air shipment",
    "tooling", "sample", "FA",
    "partial", "balance",
]
