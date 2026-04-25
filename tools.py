"""
Tool layer for external API calls.

Layer 4 rule:
- This module owns API integration only.
- Local validators run before process_payment.
- All API, timeout, network, and validation failures are normalized to ApiError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests

from config import PAYMENT_API_BASE, REQUEST_TIMEOUT
from validators import (
    validate_account_id,
    validate_amount,
    validate_card_number,
    validate_cvv,
    validate_expiry,
    validate_full_name,
)


@dataclass(frozen=True)
class Account:
    account_id: str
    full_name: str
    dob: str
    aadhaar_last4: str
    pincode: str
    balance: float


@dataclass(frozen=True)
class PaymentResult:
    success: bool
    transaction_id: Optional[str] = None
    error_code: Optional[str] = None


class ApiError(Exception):
    """
    Normalized API/tool-layer exception.

    error_code examples:
    - account_not_found
    - invalid_amount
    - insufficient_balance
    - invalid_card
    - invalid_cvv
    - invalid_expiry
    - timeout
    - network_error
    - invalid_response
    - unexpected_api_error
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(message)


def _base_url() -> str:
    base = PAYMENT_API_BASE.rstrip("/")

    # Assignment gives the OpenAPI/docs base, but actual API routes live at /api.
    if base.endswith("/openapi"):
        base = base[: -len("/openapi")]

    return base


def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{_base_url()}{endpoint}"

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.Timeout as exc:
        raise ApiError(
            "timeout",
            "The payment service timed out.",
            retryable=True,
        ) from exc
    except requests.RequestException as exc:
        raise ApiError(
            "network_error",
            "Could not reach the payment service.",
            retryable=True,
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise ApiError(
            "invalid_response",
            "The payment service returned a non-JSON response.",
            status_code=response.status_code,
            retryable=False,
        ) from exc

    if response.status_code in (404, 422):
        error_code = data.get("error_code") or "api_error"
        message = data.get("message") or error_code
        raise ApiError(
            error_code,
            message,
            status_code=response.status_code,
            retryable=response.status_code == 422,
        )

    if response.status_code >= 500:
        raise ApiError(
            "server_error",
            "The payment service is temporarily unavailable.",
            status_code=response.status_code,
            retryable=True,
        )

    if response.status_code >= 400:
        error_code = data.get("error_code") or "unexpected_api_error"
        message = data.get("message") or "The payment service rejected the request."
        raise ApiError(
            error_code,
            message,
            status_code=response.status_code,
            retryable=False,
        )

    return data


def lookup_account(account_id: str) -> Account:
    """
    Look up account details by account ID.

    Raises:
        ApiError: for invalid input, 404, timeout, network, or malformed response.
    """
    clean_account_id = validate_account_id(account_id)

    if not clean_account_id:
        raise ApiError(
            "invalid_account_id",
            "Account ID must look like ACC followed by digits.",
            retryable=False,
        )

    data = _post("/api/lookup-account", {"account_id": clean_account_id})

    required_fields = [
        "account_id",
        "full_name",
        "dob",
        "aadhaar_last4",
        "pincode",
        "balance",
    ]

    if not all(field in data for field in required_fields):
        raise ApiError(
            "invalid_response",
            "Account lookup response is missing required fields.",
            retryable=False,
        )

    try:
        return Account(
            account_id=str(data["account_id"]),
            full_name=str(data["full_name"]),
            dob=str(data["dob"]),
            aadhaar_last4=str(data["aadhaar_last4"]),
            pincode=str(data["pincode"]),
            balance=float(data["balance"]),
        )
    except (TypeError, ValueError) as exc:
        raise ApiError(
            "invalid_response",
            "Account lookup response has invalid field types.",
            retryable=False,
        ) from exc


def validate_payment_inputs(
    *,
    account_id: str,
    amount: Any,
    cardholder_name: str,
    card_number: str,
    cvv: str,
    expiry_month: Any,
    expiry_year: Any,
) -> dict[str, Any]:
    """
    Validate and normalize payment inputs before the API call.
    """
    clean_account_id = validate_account_id(account_id)
    if not clean_account_id:
        raise ApiError(
            "invalid_account_id",
            "Account ID is invalid.",
            retryable=False,
        )

    clean_amount = validate_amount(amount)
    if clean_amount is None:
        raise ApiError(
            "invalid_amount",
            "Amount must be greater than zero and have no more than 2 decimal places.",
            retryable=True,
        )

    clean_cardholder_name = validate_full_name(cardholder_name)
    if not clean_cardholder_name:
        raise ApiError(
            "invalid_cardholder_name",
            "Cardholder name is required.",
            retryable=True,
        )

    clean_card_number = validate_card_number(card_number)
    if not clean_card_number:
        raise ApiError(
            "invalid_card",
            "Card number is invalid.",
            retryable=True,
        )

    clean_cvv = validate_cvv(cvv, card_number=clean_card_number)
    if not clean_cvv:
        raise ApiError(
            "invalid_cvv",
            "CVV is invalid.",
            retryable=True,
        )

    clean_expiry = validate_expiry(expiry_month, expiry_year)
    if not clean_expiry:
        raise ApiError(
            "invalid_expiry",
            "Expiry date is invalid or expired.",
            retryable=True,
        )

    clean_expiry_month, clean_expiry_year = clean_expiry

    return {
        "account_id": clean_account_id,
        "amount": clean_amount,
        "cardholder_name": clean_cardholder_name,
        "card_number": clean_card_number,
        "cvv": clean_cvv,
        "expiry_month": clean_expiry_month,
        "expiry_year": clean_expiry_year,
    }


def process_payment(
    *,
    account_id: str,
    amount: Any,
    cardholder_name: str,
    card_number: str,
    cvv: str,
    expiry_month: Any,
    expiry_year: Any,
) -> PaymentResult:
    """
    Process a card payment.

    Important:
    - This function assumes the policy layer has already confirmed state.verified == True.
    - This function still validates local payment fields before calling the API.
    """
    clean = validate_payment_inputs(
        account_id=account_id,
        amount=amount,
        cardholder_name=cardholder_name,
        card_number=card_number,
        cvv=cvv,
        expiry_month=expiry_month,
        expiry_year=expiry_year,
    )

    payload = {
        "account_id": clean["account_id"],
        "amount": clean["amount"],
        "payment_method": {
            "type": "card",
            "card": {
                "cardholder_name": clean["cardholder_name"],
                "card_number": clean["card_number"],
                "cvv": clean["cvv"],
                "expiry_month": clean["expiry_month"],
                "expiry_year": clean["expiry_year"],
            },
        },
    }

    data = _post("/api/process-payment", payload)

    if data.get("success") is True:
        transaction_id = data.get("transaction_id")
        if not transaction_id:
            raise ApiError(
                "invalid_response",
                "Payment success response is missing transaction ID.",
                retryable=False,
            )

        return PaymentResult(
            success=True,
            transaction_id=str(transaction_id),
            error_code=None,
        )

    if data.get("success") is False:
        error_code = data.get("error_code") or "payment_failed"
        return PaymentResult(
            success=False,
            transaction_id=None,
            error_code=str(error_code),
        )

    raise ApiError(
        "invalid_response",
        "Payment response is missing success status.",
        retryable=False,
    )