"""
Canonical response templates.

Layer 5 rule:
- Templates contain the facts and policy.
- Groq may only polish tone.
- Groq must never add policy, new facts, sensitive data, or changed amounts.
"""

from __future__ import annotations

from typing import Any, Optional

from config import GROQ_API_KEY, GROQ_MODEL
from policy import sanitize_outbound_message


TEMPLATES: dict[str, str] = {
    "greeting": (
        "Hello! I can help you with your payment. "
        "Please share your account ID to get started."
    ),
    "ask_account_id": (
        "Please share your account ID to get started."
    ),
    "invalid_account_id": (
        "That account ID does not look valid. Please share it in the format ACC followed by digits, for example ACC1001."
    ),
    "account_not_found": (
        "I could not find an account with that account ID. Please check it and try again."
    ),
    "account_lookup_failed": (
        "I could not look up the account right now. Please try again."
    ),
    "ask_full_name": (
        "Got it. Please confirm your full name."
    ),
    "ask_verification_details": (
        "Thanks. To verify your identity, please provide your full name and one verification detail: date of birth, Aadhaar last 4 digits, or pincode."
    ),
    "ask_secondary_verification": (
        "Thanks. Please provide one verification detail: date of birth, Aadhaar last 4 digits, or pincode."
    ),
    "verification_required": (
        "I need to verify your identity before discussing the balance or collecting payment details."
    ),
    "verification_failed": (
        "I could not verify those details. Please check the information and try again."
    ),
    "verification_exhausted": (
        "I could not verify your identity after multiple attempts, so I have to close this session for security."
    ),
    "verification_success": (
        "Identity verified. Your outstanding balance is ₹{balance:.2f}. How much would you like to pay today?"
    ),
    "zero_balance": (
        "Identity verified. Your outstanding balance is ₹0.00, so no payment is due right now."
    ),
    "ask_amount": (
        "How much would you like to pay today? You can make a partial payment or pay the full outstanding balance of ₹{balance:.2f}."
    ),
    "invalid_amount": (
        "Please enter a valid amount greater than ₹0.00 with no more than 2 decimal places."
    ),
    "amount_exceeds_balance": (
        "The payment amount cannot be more than the outstanding balance of ₹{balance:.2f}. Please enter a lower amount."
    ),
    "ask_payment_details": (
        "Please provide the cardholder name, card number, CVV, and expiry month and year to process the payment."
    ),
    "ask_cardholder_name": (
        "Please provide the cardholder name as it appears on the card."
    ),
    "ask_card_number": (
        "Please provide the card number."
    ),
    "ask_cvv": (
        "Please provide the CVV."
    ),
    "ask_expiry": (
        "Please provide the card expiry month and year."
    ),
    "payment_ready": (
        "Thank you. I have the required payment details and will process the payment now."
    ),
    "payment_success": (
        "Payment successful. Your transaction ID is {transaction_id}. Thank you. This session is now complete."
    ),
    "payment_failed": (
        "The payment could not be completed. Please check your payment details and try again."
    ),
    "payment_exhausted": (
        "The payment could not be completed after multiple attempts, so I have to close this session."
    ),
    "invalid_card": (
        "The card number appears to be invalid. Please check the number and try again."
    ),
    "invalid_cvv": (
        "The CVV appears to be invalid. Please check it and try again."
    ),
    "invalid_expiry": (
        "The expiry date appears to be invalid or expired. Please check the month and year and try again."
    ),
    "insufficient_balance": (
        "The payment amount is higher than the outstanding balance. Please enter a lower amount."
    ),
    "api_timeout": (
        "The payment service took too long to respond. Please try again."
    ),
    "network_error": (
        "I could not reach the payment service right now. Please try again."
    ),
    "server_error": (
        "The payment service is temporarily unavailable. Please try again later."
    ),
    "unexpected_error": (
        "Something went wrong. Please try again."
    ),
    "session_closed": (
        "This session is closed. Please start a new session if you need more help."
    ),
    "sensitive_data_blocked": (
        "For security, I cannot display sensitive verification or card details."
    ),
    "recap_success": (
        "To recap, your payment of ₹{amount:.2f} was successful. Transaction ID: {transaction_id}. Thank you."
    ),
    "recap_closed": (
        "This session has been closed for security. No payment was processed."
    ),
}


API_ERROR_TEMPLATE_MAP: dict[str, str] = {
    "account_not_found": "account_not_found",
    "invalid_account_id": "invalid_account_id",
    "invalid_amount": "invalid_amount",
    "insufficient_balance": "insufficient_balance",
    "invalid_card": "invalid_card",
    "invalid_cvv": "invalid_cvv",
    "invalid_expiry": "invalid_expiry",
    "timeout": "api_timeout",
    "network_error": "network_error",
    "server_error": "server_error",
    "invalid_response": "unexpected_error",
    "unexpected_api_error": "unexpected_error",
}


POLICY_EVENT_TEMPLATE_MAP: dict[str, str] = {
    "account_required": "ask_account_id",
    "account_not_found": "account_not_found",
    "verification_required": "verification_required",
    "verification_failed": "verification_failed",
    "verification_exhausted": "verification_exhausted",
    "payment_required": "ask_payment_details",
    "payment_blocked": "verification_required",
    "payment_failed": "payment_failed",
    "payment_exhausted": "payment_exhausted",
    "invalid_amount": "invalid_amount",
    "amount_exceeds_balance": "amount_exceeds_balance",
    "session_closed": "session_closed",
}


def render_template(template_key: str, **kwargs: Any) -> str:
    """
    Render a canonical template and apply outbound sanitization.
    """
    template = TEMPLATES.get(template_key, TEMPLATES["unexpected_error"])

    try:
        rendered = template.format(**kwargs)
    except (KeyError, ValueError):
        rendered = template

    return sanitize_outbound_message(rendered)


def template_for_api_error(error_code: str) -> str:
    return API_ERROR_TEMPLATE_MAP.get(error_code, "unexpected_error")


def template_for_policy_event(event: str) -> str:
    return POLICY_EVENT_TEMPLATE_MAP.get(event, "unexpected_error")


def response_from_api_error(error_code: str, **kwargs: Any) -> str:
    return render_template(template_for_api_error(error_code), **kwargs)


def response_from_policy_event(event: str, **kwargs: Any) -> str:
    return render_template(template_for_policy_event(event), **kwargs)


def polish_with_groq(template_message: str) -> str:
    """
    Use Groq to polish tone without changing facts or policy.

    Safety behavior:
    - If Groq is unavailable, return the original canonical template.
    - If Groq returns suspicious or empty output, return the original template.
    - The final output is sanitized before returning.
    """
    canonical = sanitize_outbound_message(template_message)

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_key_here":
        return canonical

    try:
        from groq import Groq
    except ImportError:
        return canonical

    system_prompt = (
        "You polish customer support messages for a payment collection assistant. "
        "You must preserve all facts, numbers, amounts, transaction IDs, limits, and policy exactly. "
        "Do not add new information. "
        "Do not mention or reveal DOB, Aadhaar, pincode, CVV, or full card numbers. "
        "Do not make the message longer than necessary. "
        "Return only the polished message."
    )

    user_prompt = (
        "Polish the tone of this canonical message without changing its meaning:\n\n"
        f"{canonical}"
    )

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        polished = completion.choices[0].message.content or ""
    except Exception:
        return canonical

    polished = sanitize_outbound_message(polished.strip())

    if not polished:
        return canonical

    if _looks_policy_unsafe(canonical, polished):
        return canonical

    return polished


def _looks_policy_unsafe(canonical: str, polished: str) -> bool:
    """
    Conservative guard against LLM overreach.

    Allows tone changes, blocks obvious policy/fact drift.
    """
    canonical_numbers = _extract_numbers(canonical)
    polished_numbers = _extract_numbers(polished)

    if canonical_numbers != polished_numbers:
        return True

    risky_phrases = [
        "date of birth is",
        "dob is",
        "aadhaar is",
        "aadhar is",
        "pincode is",
        "pin code is",
        "cvv is",
        "card number is",
        "you are verified even if",
        "skip verification",
        "without verification",
    ]

    lowered = polished.lower()
    return any(phrase in lowered for phrase in risky_phrases)


def _extract_numbers(text: str) -> list[str]:
    import re

    return re.findall(r"\d+(?:\.\d+)?", text or "")


def make_response(
    template_key: str,
    *,
    polish: bool = False,
    **kwargs: Any,
) -> dict[str, str]:
    """
    Return the required agent response shape.

    Agent.next() can directly return this dictionary.
    """
    message = render_template(template_key, **kwargs)

    if polish:
        message = polish_with_groq(message)

    return {"message": sanitize_outbound_message(message)}