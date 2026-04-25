"""
LangGraph engine for the Payment Collection AI Agent.

Layer 3:
- Graph-based state machine using LangGraph.
- Nodes are pure Python functions:
  extract_node, policy_node, state_router_node, tool_node, response_node.
- State persists across Agent.next() calls using MemorySaver and thread_id.

Required interface:
class Agent:
    def next(self, user_input: str) -> dict:
        return {"message": "..."}
"""

from __future__ import annotations

import json
import operator
import uuid
from dataclasses import asdict
from typing import Annotated, Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import policy
from config import PAYMENT_RETRY_LIMIT, VERIFY_RETRY_LIMIT
from policy import PolicyError
from responses import make_response, response_from_api_error, render_template, template_for_api_error
from tools import Account, ApiError, PaymentResult, lookup_account, process_payment
from validators import extract_fields, mask_card_number, remove_sensitive_fields


class ConversationState(TypedDict, total=False):
    # Current turn input
    user_input: str

    # Persistent conversation history
    history: Annotated[list[dict[str, str]], operator.add]

    # Extracted / normalized user fields
    account_id: str
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

    # Account and payment state
    account_data: dict[str, Any]
    verified: bool
    verify_attempts: int
    payment_attempts: int

    # Session state
    closed: bool
    close_reason: str

    # Graph transient fields
    extracted_fields: dict[str, Any]
    route: str
    response_key: str
    response_kwargs: dict[str, Any]
    message: str

    # Tool result / error summaries
    tool_result: dict[str, Any]
    api_error: dict[str, Any]

    # Audit fields
    last_policy_event: str


TRANSIENT_KEYS = {
    "extracted_fields",
    "route",
    "response_key",
    "response_kwargs",
    "message",
    "tool_result",
    "api_error",
}


def _balance(state: ConversationState) -> float:
    account = state.get("account_data") or {}
    return float(account.get("balance", 0.0))


def _has_account_loaded(state: ConversationState) -> bool:
    account = state.get("account_data") or {}
    return bool(account.get("account_id"))


def _has_verification_secondary(state: ConversationState) -> bool:
    return bool(state.get("dob") or state.get("aadhaar_last4") or state.get("pincode"))


def _missing_payment_fields(state: ConversationState) -> list[str]:
    required = [
        "amount",
        "cardholder_name",
        "card_number",
        "cvv",
        "expiry_month",
        "expiry_year",
    ]
    return [field for field in required if state.get(field) in (None, "", [])]


def _safe_account_dict(account: Account) -> dict[str, Any]:
    return {
        "account_id": account.account_id,
        "full_name": account.full_name,
        "dob": account.dob,
        "aadhaar_last4": account.aadhaar_last4,
        "pincode": account.pincode,
        "balance": account.balance,
    }


def _safe_api_error_dict(error: ApiError) -> dict[str, Any]:
    return {
        "error_code": error.error_code,
        "message": error.message,
        "status_code": error.status_code,
        "retryable": error.retryable,
    }


def _safe_payment_result_dict(result: PaymentResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "transaction_id": result.transaction_id,
        "error_code": result.error_code,
    }


def extract_node(state: ConversationState) -> dict[str, Any]:
    """
    Layer 1: input normalization.

    Regex runs first inside validators.extract_fields().
    Groq fallback may be used there, and every value is re-validated.
    """
    user_input = state.get("user_input", "") or ""
    extracted = extract_fields(user_input, use_llm_fallback=False)

    updates: dict[str, Any] = {
        "extracted_fields": extracted,
        "history": [{"role": "user", "content": user_input}],
        "response_key": None,
        "response_kwargs": None,
        "message": None,
        "route": None,
        "tool_result": None,
        "api_error": None,
    }

    old_account_id = state.get("account_id")
    new_account_id = extracted.get("account_id")

    if new_account_id:
        updates["account_id"] = new_account_id

        # If the user switches account IDs mid-session, reset account-specific state.
        if old_account_id and old_account_id != new_account_id:
            updates["account_data"] = None
            updates["verified"] = False
            updates["verify_attempts"] = 0
            updates["payment_attempts"] = 0
            updates["closed"] = False
            updates["close_reason"] = None

            for field in [
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
            ]:
                updates[field] = None

    for field in [
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
    ]:
        if field in extracted:
            updates[field] = extracted[field]

    return updates


def policy_node(state: ConversationState) -> dict[str, Any]:
    """
    Layer 2: strict policy gatekeeping.

    No API calls happen here.
    No LLM calls happen here.
    """
    updates: dict[str, Any] = {}

    if state.get("response_key"):
        return updates

    if state.get("closed"):
        return {
            "route": "respond",
            "response_key": "session_closed",
            "response_kwargs": {},
        }

    if not state.get("account_id"):
        user_input = (state.get("user_input") or "").strip().lower()
        is_greeting = user_input in {
            "hi",
            "hello",
            "hey",
            "hii",
            "good morning",
            "good afternoon",
            "good evening",
        }

        return {
            "route": "respond",
            "response_key": "greeting" if is_greeting else "ask_account_id",
            "response_kwargs": {},
        }

    if not _has_account_loaded(state):
        try:
            policy.can_lookup_account(state)
        except PolicyError as exc:
            return {
                "route": "respond",
                "response_key": exc.safe_message_key or "ask_account_id",
                "response_kwargs": {},
                "last_policy_event": exc.event,
            }

        return {
            "route": "lookup_account",
            "response_kwargs": {},
        }

    if state.get("verified") is not True:
        if not state.get("full_name"):
            return {
                "route": "respond",
                "response_key": "ask_full_name",
                "response_kwargs": {},
            }

        if not _has_verification_secondary(state):
            return {
                "route": "respond",
                "response_key": "ask_secondary_verification",
                "response_kwargs": {},
            }

        decision = policy.verify_identity(state)

        updates["last_policy_event"] = decision.event

        if decision.allowed:
            balance = _balance(state)

            if balance <= 0:
                updates.update(
                    {
                        "verified": True,
                        "route": "respond",
                        "response_key": "zero_balance",
                        "response_kwargs": {},
                        "closed": True,
                        "close_reason": "zero_balance",
                    }
                )
            else:
                updates.update(
                    {
                        "verified": True,
                        "route": "respond",
                        "response_key": "verification_success",
                        "response_kwargs": {"balance": balance},
                    }
                )

            return updates

        updates["verify_attempts"] = state.get("verify_attempts", 0)

        if decision.event == "verification_exhausted":
            updates.update(
                {
                    "closed": True,
                    "close_reason": "verification_exhausted",
                    "route": "respond",
                    "response_key": "verification_exhausted",
                    "response_kwargs": {},
                }
            )
        else:
            updates.update(
                {
                    "route": "respond",
                    "response_key": "verification_failed",
                    "response_kwargs": {},
                }
            )

        return updates

    if state.get("amount") is not None:
        try:
            policy.validate_payment_amount_against_balance(state)
        except PolicyError as exc:
            return {
                "route": "respond",
                "response_key": exc.safe_message_key or exc.event,
                "response_kwargs": {"balance": _balance(state)},
                "last_policy_event": exc.event,
            }

    missing_payment = _missing_payment_fields(state)

    if state.get("amount") is None:
        return {
            "route": "respond",
            "response_key": "ask_amount",
            "response_kwargs": {"balance": _balance(state)},
        }

    if missing_payment:
        if missing_payment == ["cardholder_name"]:
            key = "ask_cardholder_name"
        elif missing_payment == ["card_number"]:
            key = "ask_card_number"
        elif missing_payment == ["cvv"]:
            key = "ask_cvv"
        elif set(missing_payment).issubset({"expiry_month", "expiry_year"}):
            key = "ask_expiry"
        else:
            key = "ask_payment_details"

        return {
            "route": "respond",
            "response_key": key,
            "response_kwargs": {},
        }

    try:
        decision = policy.can_process_payment(state)
    except PolicyError as exc:
        return {
            "route": "respond",
            "response_key": exc.safe_message_key or exc.event,
            "response_kwargs": {"balance": _balance(state)},
            "last_policy_event": exc.event,
        }

    if decision.allowed:
        return {
            "route": "process_payment",
            "response_key": None,
            "response_kwargs": {},
            "last_policy_event": decision.event,
        }

    return {
        "route": "respond",
        "response_key": decision.safe_message_key or "unexpected_error",
        "response_kwargs": {},
        "last_policy_event": decision.event,
    }


def state_router_node(state: ConversationState) -> dict[str, Any]:
    """
    Layer 3 router.

    Converts current state into one of:
    - lookup_account
    - process_payment
    - respond
    """
    route = state.get("route")

    if route in {"lookup_account", "process_payment", "respond"}:
        return {"route": route}

    if state.get("response_key"):
        return {"route": "respond"}

    computed_route = policy.route_next_step(state)

    if computed_route in {"lookup_account", "process_payment"}:
        return {"route": computed_route}

    return {"route": "respond"}


def tool_node(state: ConversationState) -> dict[str, Any]:
    """
    Layer 4 tool calls.

    This node is the only place where lookup_account and process_payment are called.
    """
    route = state.get("route")
    updates: dict[str, Any] = {}

    if route == "lookup_account":
        try:
            account = lookup_account(state["account_id"])
        except ApiError as exc:
            updates.update(
                {
                    "route": "respond",
                    "response_key": template_for_api_error(exc.error_code),
                    "response_kwargs": {},
                    "api_error": _safe_api_error_dict(exc),
                }
            )

            # Keep session open for account ID correction unless the error is unexpected.
            if exc.error_code not in {"account_not_found", "invalid_account_id"}:
                updates["close_reason"] = exc.error_code

            return updates

        account_dict = _safe_account_dict(account)

        updates.update(
            {
                "account_data": account_dict,
                "verified": False,
                "tool_result": {
                    "tool": "lookup_account",
                    "success": True,
                    "account_id": account.account_id,
                },
            }
        )

        return updates

    if route == "process_payment":
        try:
            payment_result = process_payment(
                account_id=state["account_id"],
                amount=state["amount"],
                cardholder_name=state["cardholder_name"],
                card_number=state["card_number"],
                cvv=state["cvv"],
                expiry_month=state["expiry_month"],
                expiry_year=state["expiry_year"],
            )
        except ApiError as exc:
            decision = policy.handle_payment_failure(state, exc.error_code)

            updates.update(
                {
                    "route": "respond",
                    "response_key": response_from_api_error(exc.error_code),
                    "response_kwargs": {"balance": _balance(state)},
                    "payment_attempts": state.get("payment_attempts", 0),
                    "api_error": _safe_api_error_dict(exc),
                    "last_policy_event": decision.event,
                }
            )

            if decision.terminal:
                updates.update(
                    {
                        "closed": True,
                        "close_reason": "payment_exhausted",
                        "response_key": "payment_exhausted",
                    }
                )

            return updates

        updates["tool_result"] = {
            "tool": "process_payment",
            **_safe_payment_result_dict(payment_result),
        }

        if payment_result.success:
            updates.update(
                {
                    "route": "respond",
                    "response_key": "payment_success",
                    "response_kwargs": {
                        "transaction_id": payment_result.transaction_id,
                    },
                    "closed": True,
                    "close_reason": "payment_success",
                }
            )
            return updates

        decision = policy.handle_payment_failure(
            state,
            payment_result.error_code or "payment_failed",
        )

        template_key = response_from_api_error(payment_result.error_code or "payment_failed")

        updates.update(
            {
                "route": "respond",
                "response_key": template_key,
                "response_kwargs": {"balance": _balance(state)},
                "payment_attempts": state.get("payment_attempts", 0),
                "last_policy_event": decision.event,
            }
        )

        if decision.terminal:
            updates.update(
                {
                    "closed": True,
                    "close_reason": "payment_exhausted",
                    "response_key": "payment_exhausted",
                }
            )

        return updates

    return {
        "route": "respond",
        "response_key": state.get("response_key") or "unexpected_error",
        "response_kwargs": state.get("response_kwargs") or {},
    }


def response_node(state: ConversationState) -> dict[str, Any]:
    """
    Layer 5 response generation.

    Templates are canonical. Groq polishing is optional and constrained inside responses.py.
    """
    response_key = state.get("response_key") or "unexpected_error"
    kwargs = state.get("response_kwargs") or {}

    response = make_response(response_key, polish=False, **kwargs)
    message = response["message"]

    return {
        "message": message,
        "history": [{"role": "assistant", "content": message}],
    }


def _route_after_router(state: ConversationState) -> str:
    route = state.get("route")

    if route in {"lookup_account", "process_payment"}:
        return "tool"

    return "response"


def build_graph():
    graph = StateGraph(ConversationState)

    graph.add_node("extract_node", extract_node)
    graph.add_node("policy_node", policy_node)
    graph.add_node("state_router_node", state_router_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("response_node", response_node)

    graph.add_edge(START, "extract_node")
    graph.add_edge("extract_node", "policy_node")
    graph.add_edge("policy_node", "state_router_node")

    graph.add_conditional_edges(
        "state_router_node",
        _route_after_router,
        {
            "tool": "tool_node",
            "response": "response_node",
        },
    )

    # After a tool call, return to policy.
    # Example: lookup_account loads account_data, then policy decides whether to ask for
    # verification or verify already-provided out-of-order information.
    graph.add_edge("tool_node", "policy_node")

    graph.add_edge("response_node", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


class Agent:
    def __init__(self) -> None:
        self.graph = build_graph()
        self.thread_id = str(uuid.uuid4())
        self._started = False

    def next(self, user_input: str) -> dict:
        """
        Process one conversation turn.

        Args:
            user_input: user's raw message

        Returns:
            {"message": str}
        """

        if not self._started:
            graph_input = {
                "user_input": user_input,
                "verified": False,
                "closed": False,
                "verify_attempts": 0,
                "payment_attempts": 0,
                "history": [],
            }
            self._started = True
        else:
            graph_input = {
                "user_input": user_input,
            }

        result = self.graph.invoke(
            graph_input,
            config={"configurable": {"thread_id": self.thread_id}},
        )

        self._emit_structured_log(result)

        message = result.get("message") or render_template("unexpected_error")
        return {"message": message}


    def _emit_structured_log(self, state: ConversationState) -> None:
        """
        Layer 6 structured log emission.

        Rules:
        - Mask card number to last 4.
        - Omit CVV entirely.
        - Do not log DOB, Aadhaar last 4, or pincode.
        - Do not log raw conversation history.
        """
        account_data = state.get("account_data") or {}
        extracted_fields = state.get("extracted_fields") or {}

        safe_extracted = remove_sensitive_fields(dict(extracted_fields))
        if safe_extracted.get("card_number"):
            safe_extracted["card_number"] = mask_card_number(safe_extracted["card_number"])
        safe_extracted.pop("cvv", None)

        safe_payment_snapshot = {
            "amount": state.get("amount"),
            "cardholder_name_present": bool(state.get("cardholder_name")),
            "card_number": mask_card_number(state.get("card_number"))
            if state.get("card_number")
            else None,
            "expiry_present": bool(state.get("expiry_month") and state.get("expiry_year")),
            "cvv_present": bool(state.get("cvv")),
        }

        # CVV presence is okay for debugging completeness, but the value is never logged.
        safe_payment_snapshot.pop("cvv", None)

        log_record = {
            "event": "agent_turn_completed",
            "thread_id": self.thread_id,
            "route": state.get("route"),
            "response_key": state.get("response_key"),
            "last_policy_event": state.get("last_policy_event"),
            "account_id": state.get("account_id"),
            "account_loaded": bool(account_data.get("account_id")),
            "verified": bool(state.get("verified")),
            "verify_attempts": int(state.get("verify_attempts") or 0),
            "verify_retry_limit": VERIFY_RETRY_LIMIT,
            "payment_attempts": int(state.get("payment_attempts") or 0),
            "payment_retry_limit": PAYMENT_RETRY_LIMIT,
            "closed": bool(state.get("closed")),
            "close_reason": state.get("close_reason"),
            "balance_known": "balance" in account_data,
            "extracted_fields": safe_extracted,
            "payment_snapshot": safe_payment_snapshot,
            "api_error": state.get("api_error"),
            "tool_result": state.get("tool_result"),
            "history_length": len(state.get("history") or []),
        }

        print(json.dumps(log_record, ensure_ascii=False, sort_keys=True))