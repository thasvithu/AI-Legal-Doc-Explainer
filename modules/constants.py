"""Centralized constants & keyword maps.

Separated out from analysis to make future maintenance / expansion easier.
TODO: Expand categories with industry-specific taxonomies (e.g., SaaS, Lease, Employment).
"""

# Risk-related keyword -> brief guidance
RISK_KEYWORDS = {
    "penalty": "Potential penalty clause – check amounts and triggers.",
    "terminate": "Termination terms – verify notice periods and conditions.",
    "auto-renew": "Auto-renewal – ensure you know how to opt out.",
    "renew": "Renewal terms – look for automatic extensions.",
    "indemnify": "Indemnification – check scope of liability.",
    "liability": "Liability limitation – confirm caps and exclusions.",
    "warranty": "Warranty/guarantee terms – confirm duration and scope.",
    "confidential": "Confidentiality obligations – check duration & carve-outs.",
    "governing law": "Jurisdiction – ensure acceptable governing law.",
    "exclusive": "Exclusivity – may restrict other partnerships.",
    "non-compete": "Non-compete – evaluate scope & duration.",
}

# Clause categories (simple heuristic mapping)
CLAUSE_CATEGORIES = {
    "terminate": "Termination",
    "termination": "Termination",
    "penalty": "Penalty",
    "auto-renew": "Renewal",
    "renew": "Renewal",
    "indemnify": "Indemnification",
    "indemnification": "Indemnification",
    "liability": "Liability",
    "warranty": "Warranty",
    "confidential": "Confidentiality",
    "governing law": "Jurisdiction",
    "jurisdiction": "Jurisdiction",
    "exclusive": "Exclusivity",
    "non-compete": "Non-Compete",
    "payment": "Payment",
    "fee": "Payment",
    "intellectual property": "Intellectual Property",
    "ownership": "Intellectual Property",
}
