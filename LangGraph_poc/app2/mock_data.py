MOCK_EMAILS = [
    {
        "label": "New PO — Standard order from ABC Corp",
        "email": (
            "Hi,\n\n"
            "Please find attached our new purchase order PO-2026-0412.\n\n"
            "- Customer: ABC Corporation\n"
            "- Item: Industrial Filter Unit Model X200\n"
            "- Quantity: 50 units\n"
            "- Required delivery date: 2026-05-01\n"
            "- Delivery address: 123 Factory Road, Taipei\n\n"
            "Please confirm receipt and availability at your earliest convenience.\n\n"
            "Best regards,\nJohn Chen\nABC Corporation"
        ),
    },
    {
        "label": "Change Order — Quantity update from Delta Ltd",
        "email": (
            "Hello,\n\n"
            "We need to revise our existing order PO-2026-0388.\n\n"
            "Original quantity was 30 units of SKU-7890 (Hydraulic Pump).\n"
            "We'd like to increase this to 45 units due to higher-than-expected demand.\n\n"
            "Can you confirm whether you can accommodate this change and whether it affects "
            "the original delivery date of 2026-04-28?\n\n"
            "Thank you,\nSarah Liu\nDelta Ltd"
        ),
    },
    {
        "label": "Wrong Order — Incorrect item received",
        "email": (
            "Dear Support Team,\n\n"
            "We received our shipment today for order PO-2026-0375 but unfortunately the wrong "
            "items were delivered.\n\n"
            "We ordered: Pressure Valve Type A (SKU-1122), Qty: 20\n"
            "We received: Pressure Valve Type B (SKU-1133), Qty: 20\n\n"
            "This is causing a production line delay. Please advise on the fastest way to "
            "get the correct items and arrange return pickup for the wrong shipment.\n\n"
            "Regards,\nMike Wang\nSunrise Manufacturing"
        ),
    },
    {
        "label": "Wrong Order — Missing items in shipment",
        "email": (
            "Hi,\n\n"
            "Our shipment for order PO-2026-0401 arrived today but was short by 10 units.\n\n"
            "Ordered: 30 units of Cable Harness H-500\n"
            "Received: 20 units\n\n"
            "Please arrange urgent delivery of the remaining 10 units. "
            "Our assembly line is scheduled to start on 2026-04-20 and we cannot proceed without the full quantity.\n\n"
            "Best,\nAnna Tsai\nVertex Electronics"
        ),
    },
    {
        "label": "Other — General enquiry about product specs",
        "email": (
            "Hello,\n\n"
            "We are evaluating suppliers for our upcoming project and came across your product catalogue.\n\n"
            "Could you please provide detailed specifications and pricing for the following items?\n"
            "- Industrial Conveyor Belt Series C\n"
            "- Load capacity up to 500kg\n"
            "- Operating temperature range: -10°C to 60°C\n\n"
            "We would also like to know lead times for an initial order of ~100 units.\n\n"
            "Thank you,\nDavid Ho\nNova Logistics"
        ),
    },
    {
        "label": "New PO — Urgent order with bank transfer mention",
        "email": (
            "Dear Team,\n\n"
            "Please process the attached PO-2026-0420 urgently.\n\n"
            "- Item: Server Rack Unit R-900\n"
            "- Quantity: 5 units\n"
            "- Required by: 2026-04-18 (hard deadline)\n\n"
            "We will arrange bank transfer payment upon your invoice. "
            "Please send banking details along with the order confirmation.\n\n"
            "This order is subject to our NDA signed on 2025-12-01.\n\n"
            "Regards,\nTom Lin\nZenith IT Solutions"
        ),
    },
]
