"""
Automated evaluation for the Payment Collection AI Agent.

This script uses mocked API calls so evaluation is deterministic and does not
depend on the external payment API being available.

Run:
    python evaluate.py
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest.mock import patch

import agent as agent_module
from agent import Agent
from tools import Account, ApiError, PaymentResult


MOCK_ACCOUNTS: dict[str, Account] = {
    "ACC1001": Account(
        account_id="ACC1001",
        full_name="Nithin Jain",
        dob="1990-05-14",
        aadhaar_last4="4321",
        pincode="400001",
        balance=1250.75,
    ),
    "ACC1002": Account(
        account_id="ACC1002",
        full_name="Rajarajeswari Balasubramaniam",
        dob="1985-11-23",
        aadhaar_last4="9876",
        pincode="400002",
        balance=540.00,
    ),
    "ACC1003": Account(
        account_id="ACC1003",
        full_name="Priya Agarwal",
        dob="1992-08-10",
        aadhaar_last4="2468",
        pincode="400003",
        balance=0.00,
    ),
    "ACC1004": Account(
        account_id="ACC1004",
        full_name="Rahul Mehta",
        dob="1988-02-29",
        aadhaar_last4="1357",
        pincode="400004",
        balance=3200.50,
    ),
}


SENSITIVE_VALUES = [
    "1990-05-14",
    "1985-11-23",
    "1992-08-10",
    "1988-02-29",
    "4321",
    "9876",
    "2468",
    "1357",
    "400001",
    "400002",
    "400003",
    "400004",
    "123",
    "1234",
    "4532015112830366",
    "371449635398431",
]


@dataclass(frozen=True)
class TestCase:
    name: str
    turns: list[str]
    expected_substrings: list[str]
    expected_final_state: dict[str, Any]
    payment_should_be_called: bool = False
    payment_error_code: Optional[str] = None
    expected_lookup_count: Optional[int] = None


class MockPaymentBackend:
    def __init__(self, *, payment_error_code: Optional[str] = None) -> None:
        self.payment_error_code = payment_error_code
        self.lookup_calls: list[str] = []
        self.payment_calls: list[dict[str, Any]] = []

    def lookup_account(self, account_id: str) -> Account:
        self.lookup_calls.append(account_id)

        if account_id not in MOCK_ACCOUNTS:
            raise ApiError(
                "account_not_found",
                "No account found with the provided account_id.",
                status_code=404,
                retryable=False,
            )

        return MOCK_ACCOUNTS[account_id]

    def process_payment(self, **kwargs: Any) -> PaymentResult:
        self.payment_calls.append(kwargs)

        if self.payment_error_code:
            if self.payment_error_code in {
                "invalid_card",
                "invalid_cvv",
                "invalid_expiry",
                "invalid_amount",
            }:
                raise ApiError(
                    self.payment_error_code,
                    self.payment_error_code,
                    status_code=422,
                    retryable=True,
                )

            return PaymentResult(
                success=False,
                transaction_id=None,
                error_code=self.payment_error_code,
            )

        return PaymentResult(
            success=True,
            transaction_id="txn_mock_success_abcde",
            error_code=None,
        )


def check_sensitive_data_leak(text: str) -> None:
    lowered = text.lower()

    forbidden_labels = [
        "dob is",
        "date of birth is",
        "aadhaar is",
        "aadhar is",
        "pincode is",
        "pin code is",
        "cvv is",
        "card number is",
    ]

    for label in forbidden_labels:
        assert label not in lowered, f"Sensitive label leaked: {label}"

    for value in SENSITIVE_VALUES:
        # Last-4 Aadhaar values can also appear as masked card last-4 in logs,
        # so only fail if the message/log clearly exposes them as identity fields.
        if value in {"4321", "9876", "2468", "1357", "400001", "400002", "400003", "400004"}:
            identity_context = re.compile(
                rf"(aadhaar|aadhar|pincode|pin\s*code|dob|date\s*of\s*birth).{{0,30}}{re.escape(value)}",
                re.IGNORECASE,
            )
            assert not identity_context.search(text), f"Sensitive identity value leaked: {value}"
        else:
            assert value not in text, f"Sensitive value leaked: {value}"


def get_final_state(agent: Agent) -> dict[str, Any]:
    snapshot = agent.graph.get_state(
        config={"configurable": {"thread_id": agent.thread_id}}
    )
    return dict(snapshot.values)


def run_test_case(case: TestCase) -> None:
    backend = MockPaymentBackend(payment_error_code=case.payment_error_code)

    captured_stdout = io.StringIO()
    messages: list[str] = []

    with patch.object(agent_module, "lookup_account", backend.lookup_account), patch.object(
        agent_module, "process_payment", backend.process_payment
    ), contextlib.redirect_stdout(captured_stdout):
        test_agent = Agent()

        for turn in case.turns:
            response = test_agent.next(turn)
            assert isinstance(response, dict), "Agent.next() must return a dict"
            assert "message" in response, "Agent.next() must return {'message': str}"
            assert isinstance(response["message"], str), "message must be a string"
            messages.append(response["message"])

        final_state = get_final_state(test_agent)

    combined_output = "\n".join(messages) + "\n" + captured_stdout.getvalue()
    check_sensitive_data_leak(combined_output)

    for expected in case.expected_substrings:
        assert expected.lower() in combined_output.lower(), (
            f"Expected substring not found in {case.name}: {expected}\n"
            f"Messages:\n{messages}"
        )

    for key, expected_value in case.expected_final_state.items():
        actual_value = final_state.get(key)
        assert actual_value == expected_value, (
            f"{case.name}: expected final_state[{key!r}]={expected_value!r}, "
            f"got {actual_value!r}"
        )

    if case.payment_should_be_called:
        assert len(backend.payment_calls) >= 1, f"{case.name}: expected payment call"
    else:
        assert len(backend.payment_calls) == 0, f"{case.name}: payment should not be called"

    if case.expected_lookup_count is not None:
        assert len(backend.lookup_calls) == case.expected_lookup_count, (
            f"{case.name}: expected {case.expected_lookup_count} lookup calls, "
            f"got {len(backend.lookup_calls)}"
        )


TEST_CASES: list[TestCase] = [
    TestCase(
        name="01 successful full payment ACC1001 using DOB",
        turns=[
            "Hi",
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "1250.75",
            "Cardholder name Nithin Jain, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=[
            "account ID",
            "Identity verified",
            "Payment successful",
            "txn_mock_success_abcde",
        ],
        expected_final_state={"verified": True, "closed": True, "close_reason": "payment_success"},
        payment_should_be_called=True,
        expected_lookup_count=1,
    ),
    TestCase(
        name="02 successful partial payment using Aadhaar last 4",
        turns=[
            "My account ID is ACC1001",
            "My name is Nithin Jain",
            "Aadhaar last 4 is 4321",
            "Pay 500",
            "Cardholder name Nithin Jain, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["Identity verified", "Payment successful"],
        expected_final_state={"verified": True, "closed": True, "close_reason": "payment_success"},
        payment_should_be_called=True,
    ),
    TestCase(
        name="03 successful verification using pincode",
        turns=[
            "ACC1002",
            "Rajarajeswari Balasubramaniam",
            "pincode 400002",
            "100",
            "Cardholder name Rajarajeswari Balasubramaniam, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["Identity verified", "Payment successful"],
        expected_final_state={"verified": True, "closed": True},
        payment_should_be_called=True,
    ),
    TestCase(
        name="04 account not found",
        turns=["ACC9999"],
        expected_substrings=["could not find an account"],
        expected_final_state={"verified": False},
        payment_should_be_called=False,
        expected_lookup_count=1,
    ),
    TestCase(
        name="05 exact case-sensitive name mismatch blocks payment",
        turns=[
            "ACC1001",
            "nithin jain",
            "1990-05-14",
            "Pay 500",
            "Cardholder name Nithin Jain, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["could not verify"],
        expected_final_state={"verified": False},
        payment_should_be_called=False,
    ),
    TestCase(
        name="06 verification retry exhaustion closes session",
        turns=[
            "ACC1001",
            "Wrong Name",
            "1990-05-14",
            "Wrong Name",
            "4321",
            "Wrong Name",
            "400001",
        ],
        expected_substrings=["close this session"],
        expected_final_state={"verified": False, "closed": True, "close_reason": "verification_exhausted"},
        payment_should_be_called=False,
    ),
    TestCase(
        name="07 invalid non-leap DOB does not verify ACC1004",
        turns=[
            "ACC1004",
            "Rahul Mehta",
            "1990-02-29",
            "Pay 100",
            "Cardholder name Rahul Mehta, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["Please provide one verification detail"],
        expected_final_state={"verified": False},
        payment_should_be_called=False,
    ),
    TestCase(
        name="08 leap year DOB verifies ACC1004",
        turns=[
            "ACC1004",
            "Rahul Mehta",
            "1988-02-29",
            "100",
            "Cardholder name Rahul Mehta, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["Identity verified", "Payment successful"],
        expected_final_state={"verified": True, "closed": True},
        payment_should_be_called=True,
    ),
    TestCase(
        name="09 zero balance closes without payment",
        turns=[
            "ACC1003",
            "Priya Agarwal",
            "1992-08-10",
            "Pay 10",
            "Cardholder name Priya Agarwal, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["₹0.00", "no payment is due"],
        expected_final_state={"verified": True, "closed": True, "close_reason": "zero_balance"},
        payment_should_be_called=False,
    ),
    TestCase(
        name="10 amount greater than balance is blocked before API",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "Pay 2000",
            "Cardholder name Nithin Jain, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["cannot be more than the outstanding balance"],
        expected_final_state={"verified": True},
        payment_should_be_called=False,
    ),
    TestCase(
        name="11 invalid amount zero is blocked",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "Pay 0",
        ],
        expected_substrings=["valid amount"],
        expected_final_state={"verified": True},
        payment_should_be_called=False,
    ),
    TestCase(
        name="12 payment invalid card failure is handled",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "Pay 100",
            "Cardholder name Nithin Jain, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["card number appears to be invalid"],
        expected_final_state={"verified": True},
        payment_should_be_called=True,
        payment_error_code="invalid_card",
    ),
    TestCase(
        name="13 payment invalid cvv failure is handled",
        turns=[
            "ACC1001",
            "Nithin Jain",
            "1990-05-14",
            "Pay 100",
            "Cardholder name Nithin Jain, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["CVV appears to be invalid"],
        expected_final_state={"verified": True},
        payment_should_be_called=True,
        payment_error_code="invalid_cvv",
    ),
    TestCase(
        name="14 out-of-order information is retained",
        turns=[
            "My name is Nithin Jain and DOB is 1990-05-14 and I want to pay 100",
            "ACC1001",
            "Cardholder name Nithin Jain, card number 4532015112830366, cvv 123, expiry 12/2027",
        ],
        expected_substrings=["Identity verified", "Payment successful"],
        expected_final_state={"verified": True, "closed": True},
        payment_should_be_called=True,
        expected_lookup_count=1,
    ),
    TestCase(
        name="15 payment is never called before verification",
        turns=[
            "ACC1001",
            "Pay 100 with card 4532015112830366 cvv 123 expiry 12/2027 cardholder Nithin Jain",
        ],
        expected_substrings=["confirm your full name"],
        expected_final_state={"verified": False},
        payment_should_be_called=False,
    ),
]


def main() -> None:
    passed = 0
    failed = 0
    failures: list[dict[str, str]] = []

    for case in TEST_CASES:
        try:
            run_test_case(case)
            passed += 1
            print(json.dumps({"test": case.name, "status": "PASSED"}))
        except Exception as exc:
            failed += 1
            failures.append({"test": case.name, "error": str(exc)})
            print(json.dumps({"test": case.name, "status": "FAILED", "error": str(exc)}))

    summary = {
        "total": len(TEST_CASES),
        "passed": passed,
        "failed": failed,
        "success_rate": passed / len(TEST_CASES),
        "failures": failures,
    }

    print(json.dumps(summary, indent=2))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()