TEMPLATES: dict[str, dict] = {
    "New PO": {
        "required_fields": ["PO number", "delivery date", "quantity", "item description"],
        "template": """Dear [Customer Name],

Thank you for your purchase order.

We have received your order and are pleased to confirm the following:
- PO Number: [PO Number]
- Item(s): [Item Description]
- Quantity: [Quantity]
- Estimated Delivery: [Delivery Date]

Please confirm the above details are correct. If you have any questions or changes, do not hesitate to contact us.

Best regards,
[Your Name]
[Company]""",
    },
    "Change Order": {
        "required_fields": ["original PO number", "change details", "updated delivery date"],
        "template": """Dear [Customer Name],

Thank you for reaching out regarding your order change request.

We have noted the following changes to your order:
- Original PO Number: [Original PO Number]
- Requested Change: [Change Details]
- Updated Delivery Date: [Updated Delivery Date]

We will process this change and confirm availability shortly. Please allow us [X] business days to review.

Best regards,
[Your Name]
[Company]""",
    },
    "Wrong Order": {
        "required_fields": ["original order number", "incorrect item received", "correct item expected"],
        "template": """Dear [Customer Name],

We sincerely apologize for the inconvenience caused by the incorrect shipment.

We have logged your case with the following details:
- Original Order Number: [Order Number]
- Item Received: [Incorrect Item]
- Item Expected: [Correct Item]

We will arrange for the correct item to be dispatched and coordinate the return of the incorrect shipment. Our logistics team will contact you within [X] business days.

Once again, we apologize for this error and appreciate your patience.

Best regards,
[Your Name]
[Company]""",
    },
    "Other": {
        "required_fields": [],
        "template": """Dear [Customer Name],

Thank you for contacting us.

We have received your message and will review it carefully. A member of our team will follow up with you within [X] business days.

If your matter is urgent, please do not hesitate to call us directly.

Best regards,
[Your Name]
[Company]""",
    },
}

RISK_KEYWORDS = [
    "bank account", "bank transfer", "wire transfer",
    "NDA", "non-disclosure", "confidential",
    "pricing", "discount", "contract",
    "legal", "dispute", "complaint",
]
