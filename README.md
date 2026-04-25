# Payment Collection AI Agent

## Overview

This project implements a production-style conversational AI agent for payment collection. The agent guides a user through account lookup, identity verification, outstanding balance disclosure, card payment collection, payment processing, and session closure.

The system is designed around strict financial-compliance rules: verification is required before payment, sensitive account data is never exposed, payment details are validated before API calls, and all state transitions are controlled through a LangGraph state machine.

The required interface is implemented exactly:

```python
class Agent:
    def next(self, user_input: str) -> dict:
        return {"message": "..."}
```

Each call to `next()` represents one user turn, and the agent maintains conversation state internally across turns.

---

## Features

- Multi-turn conversational state using LangGraph
- Required `Agent.next(user_input: str) -> dict` interface
- Account lookup through the provided payment API
- Strict identity verification before payment
- Exact and case-sensitive full name matching
- Verification using DOB, Aadhaar last 4, or pincode
- Leap-year DOB validation
- Payment amount validation
- Local card validation before payment API calls
- Retry limits for verification and payment
- Session closure after retry exhaustion
- Canonical response templates
- Optional Groq hooks for fallback extraction and tone polishing
- Structured JSON logs with sensitive data masking
- Deterministic evaluation with 15 test cases

---

## Architecture

The project follows a strict 6-layer architecture.

### Layer 1: Input Normalization — `validators.py`

This layer extracts and validates structured fields from user input.

Fields handled:

- `account_id`
- `full_name`
- `dob`
- `aadhaar_last4`
- `pincode`
- `amount`
- `cardholder_name`
- `card_number`
- `cvv`
- `expiry_month`
- `expiry_year`

Regex extraction runs first for every field. A Groq fallback extraction function is implemented, but the evaluated path uses deterministic regex extraction to preserve exact matching and predictable behavior.

### Layer 2: Policy / Guard Layer — `policy.py`

This layer is pure Python and contains no LLM calls or API calls.

It enforces:

- Account must exist before verification
- Verification retry limit
- Payment retry limit
- Payment blocked before successful verification
- Amount must be greater than 0
- Amount must be less than or equal to the outstanding balance
- Sensitive data must not appear in outbound messages
- Session closes after retry exhaustion

### Layer 3: Graph State Machine — `agent.py`

The main engine is implemented using LangGraph.

The graph uses:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
```

Graph nodes:

- `extract_node`
- `policy_node`
- `state_router_node`
- `tool_node`
- `response_node`

The graph is compiled with `MemorySaver`, and each `Agent` instance uses a unique `thread_id`. This allows state to persist across multiple calls to `Agent.next()`.

### Layer 4: Tool Layer — `tools.py`

This layer handles external API calls using `requests`.

Implemented tools:

- `lookup_account(account_id)`
- `process_payment(...)`

It also defines:

- `Account`
- `PaymentResult`
- `ApiError`

All timeouts, network failures, 404 errors, 422 errors, server errors, and unexpected API responses are normalized into typed `ApiError` exceptions.

Before calling the payment API, local validation checks:

- amount
- cardholder name
- card number
- CVV
- expiry month and year

### Layer 5: Response Layer — `responses.py`

This layer contains canonical templates for every user-facing event.

Examples:

- ask account ID
- account not found
- ask verification details
- verification success
- verification failure
- retry exhaustion
- ask payment details
- invalid amount
- invalid card
- invalid CVV
- invalid expiry
- payment success
- session closed

A Groq tone-polishing function is implemented, but templates remain the source of truth. The LLM is not allowed to change facts, policy, amounts, transaction IDs, or sensitive data rules.

### Layer 6: Observability / Evaluation — `evaluate.py`

The agent emits structured JSON logs after each turn.

Logs include:

- route
- response key
- policy event
- account loaded status
- verification status
- retry counts
- payment attempt count
- closed status
- API error summary
- tool result summary
- history length

Logs do not include:

- DOB
- Aadhaar last 4
- pincode
- CVV
- full card number
- raw conversation history

Card numbers are masked to the last 4 digits only.

---

## Project Structure

```text
payment-collection-agent/
├── agent.py
├── config.py
├── validators.py
├── policy.py
├── tools.py
├── responses.py
├── evaluate.py
├── cli.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_REPO_URL
cd payment-collection-agent
```

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama3-70b-8192
```

The agent can still run without Groq because deterministic regex extraction and canonical response templates are used by default.

Make sure `.env` is included in `.gitignore` and is not pushed to GitHub.

---

## Run Instructions

Run the interactive CLI:

```bash
python cli.py
```

Example:

```text
Payment Collection AI Agent
Type 'exit' or 'quit' to end.

You: Hi
Agent: Please share your account ID to get started.
```

To exit:

```text
exit
```

---

## Evaluation Instructions

Run the automated evaluator:

```bash
python evaluate.py
```

Expected result:

```text
15 passed, 0 failed
success_rate: 1.0
```

The evaluator uses mocked API data so tests are deterministic and do not depend on the external API being available.

The 15 test cases cover:

1. Successful full payment
2. Successful partial payment
3. Verification using pincode
4. Account not found
5. Exact case-sensitive name mismatch
6. Verification retry exhaustion
7. Invalid non-leap DOB
8. Valid leap-year DOB
9. Zero-balance account
10. Amount greater than balance
11. Invalid zero amount
12. Invalid card failure
13. Invalid CVV failure
14. Out-of-order information handling
15. Ensuring payment is never called before verification

The evaluator also checks that sensitive data is not leaked in user-facing messages or structured logs.

---

## Sample Conversations

### 1. Successful End-to-End Payment

```text
User: Hi
Agent: Hello! I can help you with your payment. Please share your account ID to get started.

User: ACC1001
Agent: Got it. Please confirm your full name.

User: Nithin Jain
Agent: Thanks. Please provide one verification detail: date of birth, Aadhaar last 4 digits, or pincode.

User: 1990-05-14
Agent: Identity verified. Your outstanding balance is ₹1250.75. How much would you like to pay today?

User: 500
Agent: Please provide the cardholder name, card number, CVV, and expiry month and year to process the payment.

User: Cardholder name Nithin Jain, card number 4532015112830366, cvv 123, expiry 12/2027
Agent: Payment successful. Your transaction ID is txn_mock_success_abcde. Thank you. This session is now complete.
```

### 2. Verification Failure: User Exhausts Retries

```text
User: ACC1001
Agent: Got it. Please confirm your full name.

User: Wrong Name
Agent: Thanks. Please provide one verification detail: date of birth, Aadhaar last 4 digits, or pincode.

User: 1990-05-14
Agent: I could not verify those details. Please check the information and try again.

User: Wrong Name
Agent: I could not verify those details. Please check the information and try again.

User: Wrong Name
Agent: I could not verify your identity after multiple attempts, so I have to close this session for security.
```

### 3. Payment Failure: Invalid Card

```text
User: ACC1001
Agent: Got it. Please confirm your full name.

User: Nithin Jain
Agent: Thanks. Please provide one verification detail: date of birth, Aadhaar last 4 digits, or pincode.

User: 1990-05-14
Agent: Identity verified. Your outstanding balance is ₹1250.75. How much would you like to pay today?

User: 100
Agent: Please provide the cardholder name, card number, CVV, and expiry month and year to process the payment.

User: Cardholder name Nithin Jain, card number 4111111111111112, cvv 123, expiry 12/2027
Agent: The card number appears to be invalid. Please check the number and try again.
```

### 4. Edge Case: Leap Year DOB

```text
User: ACC1004
Agent: Got it. Please confirm your full name.

User: Rahul Mehta
Agent: Thanks. Please provide one verification detail: date of birth, Aadhaar last 4 digits, or pincode.

User: 1988-02-29
Agent: Identity verified. Your outstanding balance is ₹3200.50. How much would you like to pay today?
```

Invalid DOB example:

```text
User: ACC1004
Agent: Got it. Please confirm your full name.

User: Rahul Mehta
Agent: Thanks. Please provide one verification detail: date of birth, Aadhaar last 4 digits, or pincode.

User: 1990-02-29
Agent: Thanks. Please provide one verification detail: date of birth, Aadhaar last 4 digits, or pincode.
```

`1988-02-29` is accepted because 1988 is a leap year. `1990-02-29` is rejected during validation because 1990 is not a leap year, and it does not count as a verification attempt.

---

## Design Document

### Architecture Overview

The agent is built using a strict 6-layer architecture to separate extraction, policy enforcement, state management, external tools, response generation, and observability.

The main engine is implemented in `agent.py` using LangGraph. Each call to `Agent.next()` passes the user message through extraction, policy checks, routing, optional tool execution, and response generation. The graph is compiled with `MemorySaver`, which allows conversation state to persist across turns.

The supporting modules are intentionally separated:

- `validators.py` handles input extraction and validation.
- `policy.py` enforces business and security rules.
- `tools.py` handles external API calls.
- `responses.py` generates safe user-facing messages.
- `evaluate.py` tests the agent with mocked API data.

The graph flow is:

```text
START
  → extract_node
  → policy_node
  → state_router_node
  → tool_node or response_node
  → END
```

After a tool call, the graph returns to the policy node so the next state is decided consistently.

### Key Decisions Made

#### Regex-first extraction

The most important design decision was to use regex-first extraction instead of LLM-first extraction.

This is a financial compliance workflow, and most required fields are structured. Account IDs, DOBs, pincodes, card numbers, CVVs, expiry dates, and payment amounts all have predictable patterns. Regex extraction is deterministic, testable, and safer than relying on an LLM for critical fields.

A Groq fallback extraction function is implemented in `validators.py`, but the evaluated agent keeps it disabled by default. This prevents the LLM from accidentally changing identity-sensitive values.

For example, verification requires exact and case-sensitive name matching. If a user enters `nithin jain`, the system must not treat it as `Nithin Jain`. An LLM could normalize casing, so the production path keeps identity extraction deterministic.

#### Rule-based verification

Verification is fully rule-based and implemented in Python.

The user is verified only if:

```text
Full name matches exactly AND at least one secondary factor matches exactly.
```

The secondary factor can be:

```text
DOB OR Aadhaar last 4 OR pincode
```

This logic is not handled by the LLM because identity verification must be deterministic and auditable.

#### Templates-first responses

All user-facing messages come from canonical templates in `responses.py`.

This prevents the assistant from accidentally:

- revealing sensitive account data
- changing verification rules
- inventing payment facts
- changing retry limits
- adding unsupported policy

A Groq tone-polishing function is implemented, but templates remain the source of truth.

#### LangGraph state machine

LangGraph was chosen because the payment flow is naturally a state machine. The agent must know whether it is waiting for account ID, verification details, payment amount, card details, or payment result.

Using LangGraph makes transitions explicit and testable. It also ensures that tools are called only at the right time.

#### MemorySaver for state persistence

The assignment requires this interface:

```python
agent.next(user_input: str) -> dict
```

Each call is one user turn, but the conversation state must persist across calls. To support this, the graph is compiled with `MemorySaver`, and each `Agent` instance uses a unique `thread_id`.

### Tradeoffs Accepted

#### LLM hooks are implemented but disabled by default

The LLM is integrated only in the two permitted layers:

- Layer 1: fallback extraction in `validators.py`
- Layer 5: tone polishing in `responses.py`

However, both are disabled in the evaluated flow by default. This makes the system more deterministic and safer, but less flexible for unusual phrasing.

This tradeoff is intentional because the workflow involves identity verification and payment processing.

#### In-memory state

The project uses LangGraph `MemorySaver`, which is appropriate for the assignment and local evaluation. A production system should use durable storage such as Redis, Postgres, or another persistent checkpoint store.

#### Regex extraction limitations

Regex extraction is reliable for structured data but may miss unusual natural-language inputs. A production version could selectively enable LLM fallback for low-risk fields while keeping identity and payment validation deterministic.

#### Simple CLI

The CLI is intentionally simple. It demonstrates the required `Agent.next()` interface but is not a production chat UI.

### What I Would Improve With More Time

With more time, I would add:

1. Durable checkpoint storage using Redis or Postgres
2. A FastAPI wrapper around the `Agent.next()` interface
3. Stronger PCI-style logging controls
4. More adversarial tests for prompt injection
5. More robust natural-language parsing for unusual phrasing
6. Field-level controls for enabling LLM fallback only on safe fields
7. Monitoring for verification failure rate, retry exhaustion rate, payment failure reasons, and API latency
8. More detailed structured metrics for tool calls and policy decisions
9. A better user-facing recovery flow for retryable card failures
10. Production-grade observability with log aggregation and dashboards

---

## Security and Compliance Notes

The system follows these security rules:

- Payment is blocked until identity verification succeeds.
- Name matching is exact and case-sensitive.
- DOB, Aadhaar last 4, and pincode are never revealed back to the user.
- CVV is never logged.
- Full card number is never logged.
- Card number is masked to the last 4 digits in logs.
- Invalid DOBs do not count as verification attempts.
- Sessions close after retry exhaustion.
- Structured logs omit raw conversation history.
- The LLM is not allowed to make policy decisions.
- The LLM is not allowed to decide whether a user is verified.
- The LLM is not allowed to alter payment facts or transaction IDs.

---

## Test Accounts Used in Evaluation

The evaluation script uses mocked versions of the assignment-provided accounts:

| Account ID | Full Name | DOB | Aadhaar Last 4 | Pincode | Balance |
|---|---|---|---|---|---:|
| ACC1001 | Nithin Jain | 1990-05-14 | 4321 | 400001 | ₹1250.75 |
| ACC1002 | Rajarajeswari Balasubramaniam | 1985-11-23 | 9876 | 400002 | ₹540.00 |
| ACC1003 | Priya Agarwal | 1992-08-10 | 2468 | 400003 | ₹0.00 |
| ACC1004 | Rahul Mehta | 1988-02-29 | 1357 | 400004 | ₹3200.50 |

---

## Final Validation

Before submitting, run:

```bash
python evaluate.py
```

Expected output:

```text
{
  "total": 15,
  "passed": 15,
  "failed": 0,
  "success_rate": 1.0,
  "failures": []
}
```

Then run the CLI manually:

```bash
python cli.py
```

This confirms both automated and interactive flows work.

---

## Future Improvements

- Durable LangGraph checkpoint storage
- FastAPI deployment wrapper
- More robust natural-language parsing
- More adversarial security tests
- Production-grade monitoring and alerting
- PCI-focused audit logging
- Better payment retry recovery
- Field-level LLM fallback controls
