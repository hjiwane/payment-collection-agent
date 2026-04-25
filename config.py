"""
Configuration for the Payment Collection AI Agent.

This module intentionally contains only constants and lightweight helpers.
No business logic should live here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


PAYMENT_API_BASE = (
    "https://se-payment-verification-api.service.external.usea2.aws.prodigaltech.com/openapi"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_key_here")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

VERIFY_RETRY_LIMIT = 3
PAYMENT_RETRY_LIMIT = 3
REQUEST_TIMEOUT = 10


VERIFICATION_FIELDS = [
    "full_name",
    "dob",
    "aadhaar_last4",
    "pincode",
]

PAYMENT_FIELDS = [
    "amount",
    "cardholder_name",
    "card_number",
    "cvv",
    "expiry_month",
    "expiry_year",
]

EXTRACTABLE_FIELDS = [
    "account_id",
    "full_name",
    "dob",
    "aadhaar_last4",
    "pincode",
    "amount",
    "cardholder_name",
    "card_number",
    "cvv",
    "expiry_month",
    "expiry_year",
]

SENSITIVE_FIELDS = [
    "dob",
    "aadhaar_last4",
    "pincode",
    "cvv",
    "card_number",
]

NEVER_EXPOSE_FIELDS = [
    "dob",
    "aadhaar_last4",
    "pincode",
    "cvv",
]

CARD_MASK_PREFIX = "**** **** **** "


def get_groq_client_config() -> dict:
    """
    Return Groq configuration in a simple dictionary.

    The actual Groq client should be instantiated inside the module that needs it.
    """
    return {
        "api_key": GROQ_API_KEY,
        "model": GROQ_MODEL,
    }