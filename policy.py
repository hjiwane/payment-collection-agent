"""
Pure Python policy and guard layer.

Layer 2 rule:
- No LLM usage.
- No API calls.
- Strictly enforces verification, retry limits, payment eligibility,
  session closure, and outbound message safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, TypedDict

from config import PAYMENT_RETRY_LIMIT, VERIFY_RETRY_LIMIT
from validators import mask_card_number, remove_sensitive_fields, validate_amount


PolicyEvent = Literal[
    "allowed",
    "blocked",
    "account_required",
    "account_not_found",
    "verification_required",
    "verification_failed",
    "verification_exhausted",
    "payment_required",
    "payment_blocked",
    "payment_failed",
    "payment_exhausted",
    "invalid_amount",
    "amount_exceeds_balance",
    "session_closed",
]


class PolicyError(Exception):
    """
    Typed exception raised when a policy rule blocks execution.
    """

    def __init__(
        self,
        event: PolicyEvent,
        message: str,
        *,
        terminal: bool = False,
        safe_message_key: Optional[str] = None,
    ) -> None:
        self.event = event
        self.message = message
        self.terminal = terminal
        self.safe_message_key = safe_message_key or event
        super().__init__(message)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    event: PolicyEvent
    reason: str = ""
    terminal: bool = False
    safe_message_key: Optional[str] = None


class ConversationState(TypedDict, total=False):
    account_id: str
    account_data: dict[str, Any]
    verified: bool
    verify_attempts: int
    payment_attempts: int
    closed: bool
    close_reason: str

    full_name: str
    dob: str
    aadhaar_last4: str
    pincode: str

    amount: float
    cardholder_name: str
    card_number: str
    cvv: str
    expiry_month: int
    expiry_year: int

    last_policy_event: str


def _get_attempts(state: ConversationState, key: str) -> int:
    value = state.get(key, 0)
    return int(value or 0)


def _set_event(state: ConversationState, event: PolicyEvent) -> None:
    state["last_policy_event"] = event


def close_session(state: ConversationState, reason: str) -> ConversationState:
    state["closed"] = True
    state["close_reason"] = reason
    _set_event(state, "session_closed")
    return state


def ensure_session_open(state: ConversationState) -> PolicyDecision:
    if state.get("closed"):
        raise PolicyError(
            "session_closed",
            "Session is already closed.",
            terminal=True,
            safe_message_key="session_closed",
        )

    return PolicyDecision(True, "allowed")


def can_lookup_account(state: ConversationState) -> PolicyDecision:
    ensure_session_open(state)

    if not state.get("account_id"):
        raise PolicyError(
            "account_required",
            "Account ID is required before lookup.",
            safe_message_key="ask_account_id",
        )

    return PolicyDecision(True, "allowed")


def mark_account_loaded(
    state: ConversationState,
    account_data: dict[str, Any],
) -> ConversationState:
    if not account_data or not account_data.get("account_id"):
        raise PolicyError(
            "account_not_found",
            "Account data is missing or invalid.",
            safe_message_key="account_not_found",
        )

    state["account_data"] = dict(account_data)
    state["account_id"] = account_data["account_id"]
    state["verified"] = False
    _set_event(state, "allowed")
    return state


def account_exists(state: ConversationState) -> bool:
    return bool(state.get("account_data") and state["account_data"].get("account_id"))


def can_attempt_verification(state: ConversationState) -> PolicyDecision:
    ensure_session_open(state)

    if not account_exists(state):
        raise PolicyError(
            "account_required",
            "Account must be loaded before verification.",
            safe_message_key="ask_account_id",
        )

    if state.get("verified") is True:
        return PolicyDecision(True, "allowed", "User is already verified.")

    if _get_attempts(state, "verify_attempts") >= VERIFY_RETRY_LIMIT:
        close_session(state, "verification_exhausted")
        raise PolicyError(
            "verification_exhausted",
            "Verification retry limit exhausted.",
            terminal=True,
            safe_message_key="verification_exhausted",
        )

    return PolicyDecision(True, "allowed")


def has_minimum_verification_inputs(state: ConversationState) -> bool:
    """
    Verification requires full_name plus at least one secondary factor.

    Invalid DOBs are dropped by validators before policy receives state.
    Therefore, invalid DOB input will not count as a verification attempt unless
    a valid full_name and another valid secondary field are present.
    """
    if not state.get("full_name"):
        return False

    return bool(
        state.get("dob")
        or state.get("aadhaar_last4")
        or state.get("pincode")
    )


def verify_identity(state: ConversationState) -> PolicyDecision:
    """
    Strict verification:
    - full_name must match exactly and case-sensitively.
    - At least one secondary factor must match exactly.
    """
    can_attempt_verification(state)

    if not has_minimum_verification_inputs(state):
        return PolicyDecision(
            False,
            "verification_required",
            "Need full name and at least one valid secondary verification field.",
            safe_message_key="ask_verification_details",
        )

    account = state["account_data"]

    name_matches = state.get("full_name") == account.get("full_name")

    secondary_matches = any(
        [
            state.get("dob") is not None and state.get("dob") == account.get("dob"),
            state.get("aadhaar_last4") is not None
            and state.get("aadhaar_last4") == account.get("aadhaar_last4"),
            state.get("pincode") is not None
            and state.get("pincode") == account.get("pincode"),
        ]
    )

    if name_matches and secondary_matches:
        mark_verified(state)
        return PolicyDecision(
            True,
            "allowed",
            "Identity verified.",
            safe_message_key="verification_success",
        )

    return handle_verification_failure(state)


def mark_verified(state: ConversationState) -> ConversationState:
    if not account_exists(state):
        raise PolicyError(
            "account_required",
            "Cannot verify user without a loaded account.",
            safe_message_key="ask_account_id",
        )

    state["verified"] = True
    _set_event(state, "allowed")
    return state


def is_verification_exhausted(state: ConversationState) -> bool:
    return _get_attempts(state, "verify_attempts") >= VERIFY_RETRY_LIMIT


def handle_verification_failure(state: ConversationState) -> PolicyDecision:
    attempts = _get_attempts(state, "verify_attempts") + 1
    state["verify_attempts"] = attempts

    if attempts >= VERIFY_RETRY_LIMIT:
        close_session(state, "verification_exhausted")
        return PolicyDecision(
            False,
            "verification_exhausted",
            "Verification retry limit exhausted.",
            terminal=True,
            safe_message_key="verification_exhausted",
        )

    _set_event(state, "verification_failed")
    return PolicyDecision(
        False,
        "verification_failed",
        "Verification failed.",
        safe_message_key="verification_failed",
    )


def can_collect_payment(state: ConversationState) -> PolicyDecision:
    ensure_session_open(state)

    if state.get("verified") is not True:
        raise PolicyError(
            "verification_required",
            "Payment details cannot be collected before verification.",
            safe_message_key="verification_required",
        )

    return PolicyDecision(True, "allowed")


def validate_payment_amount_against_balance(state: ConversationState) -> PolicyDecision:
    can_collect_payment(state)

    amount = validate_amount(state.get("amount"))
    if amount is None:
        raise PolicyError(
            "invalid_amount",
            "Payment amount must be greater than 0 and have no more than 2 decimals.",
            safe_message_key="invalid_amount",
        )

    account = state.get("account_data") or {}
    balance = float(account.get("balance", 0))

    if amount > balance:
        raise PolicyError(
            "amount_exceeds_balance",
            "Payment amount cannot exceed outstanding balance.",
            safe_message_key="amount_exceeds_balance",
        )

    state["amount"] = amount

    return PolicyDecision(True, "allowed")


def has_required_payment_fields(state: ConversationState) -> bool:
    required = [
        "amount",
        "cardholder_name",
        "card_number",
        "cvv",
        "expiry_month",
        "expiry_year",
    ]

    return all(state.get(field) not in (None, "", []) for field in required)


def can_process_payment(state: ConversationState) -> PolicyDecision:
    """
    Hard payment gate.

    This must pass before tools.process_payment can ever be called.
    """
    ensure_session_open(state)

    if state.get("verified") is not True:
        raise PolicyError(
            "verification_required",
            "Never call process_payment before state.verified == True.",
            safe_message_key="verification_required",
        )

    if _get_attempts(state, "payment_attempts") >= PAYMENT_RETRY_LIMIT:
        close_session(state, "payment_exhausted")
        raise PolicyError(
            "payment_exhausted",
            "Payment retry limit exhausted.",
            terminal=True,
            safe_message_key="payment_exhausted",
        )

    if not has_required_payment_fields(state):
        return PolicyDecision(
            False,
            "payment_required",
            "Missing required payment fields.",
            safe_message_key="ask_payment_details",
        )

    validate_payment_amount_against_balance(state)

    return PolicyDecision(
        True,
        "allowed",
        "Payment may be processed.",
        safe_message_key="payment_ready",
    )


def is_payment_exhausted(state: ConversationState) -> bool:
    return _get_attempts(state, "payment_attempts") >= PAYMENT_RETRY_LIMIT


def handle_payment_failure(
    state: ConversationState,
    reason: str = "payment_failed",
) -> PolicyDecision:
    ensure_session_open(state)

    attempts = _get_attempts(state, "payment_attempts") + 1
    state["payment_attempts"] = attempts

    if attempts >= PAYMENT_RETRY_LIMIT:
        close_session(state, "payment_exhausted")
        return PolicyDecision(
            False,
            "payment_exhausted",
            reason,
            terminal=True,
            safe_message_key="payment_exhausted",
        )

    _set_event(state, "payment_failed")
    return PolicyDecision(
        False,
        "payment_failed",
        reason,
        safe_message_key="payment_failed",
    )


def route_next_step(state: ConversationState) -> str:
    """
    Deterministic state router for agent.py.

    Returns a symbolic route. The LangGraph layer will use this to decide
    the next node.
    """
    if state.get("closed"):
        return "closed"

    if not state.get("account_id"):
        return "ask_account_id"

    if not account_exists(state):
        return "lookup_account"

    if state.get("verified") is not True:
        if has_minimum_verification_inputs(state):
            return "verify_identity"
        return "ask_verification"

    if not state.get("amount"):
        return "ask_amount"

    if not has_required_payment_fields(state):
        return "ask_payment_details"

    return "process_payment"


def sanitize_outbound_message(message: str) -> str:
    """
    Remove sensitive labels and obvious sensitive values from outbound text.

    This is a final safety net. Response templates should already avoid
    sensitive data entirely.
    """
    if not message:
        return ""

    sanitized = str(message)

    sensitive_patterns = [
        r"\b(?:dob|date\s*of\s*birth)\s*[:=-]?\s*\d{4}-\d{2}-\d{2}\b",
        r"\b(?:aadhaar|aadhar|adhar)(?:\s*last\s*4)?\s*[:=-]?\s*\d{4}\b",
        r"\b(?:pincode|pin\s*code|postal\s*code|zip)\s*[:=-]?\s*\d{6}\b",
        r"\b(?:cvv|cvc|security\s*code)\s*[:=-]?\s*\d{3,4}\b",
    ]

    import re

    for pattern in sensitive_patterns:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

    return sanitized


def assert_no_sensitive_data(message: str) -> None:
    sanitized = sanitize_outbound_message(message)

    if sanitized != message:
        raise PolicyError(
            "blocked",
            "Outbound message contains sensitive data.",
            safe_message_key="sensitive_data_blocked",
        )


def mask_payment_data_for_logs(data: dict[str, Any]) -> dict[str, Any]:
    """
    Logs may include masked card last 4 only.
    CVV is omitted entirely.
    """
    safe = remove_sensitive_fields(dict(data))

    if data.get("card_number"):
        safe["card_number"] = mask_card_number(data["card_number"])

    safe.pop("cvv", None)
    return safe


def build_safe_payment_payload_preview(state: ConversationState) -> dict[str, Any]:
    """
    Returns a safe preview suitable for debugging/logging.

    This is not the API payload. The API payload belongs to tools.py.
    """
    preview = {
        "account_id": state.get("account_id"),
        "amount": state.get("amount"),
        "cardholder_name": state.get("cardholder_name"),
        "card_number": state.get("card_number"),
        "expiry_month": state.get("expiry_month"),
        "expiry_year": state.get("expiry_year"),
    }

    return mask_payment_data_for_logs(preview)